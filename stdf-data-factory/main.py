#!/usr/bin/env python3

"""
Streamlit UI for Enhanced Data Extractor
Functional programming approach - UI separated from core logic
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import time

# Import our core data extraction module
from data_extractor import (
    extract_data,
    validate_query,
    validate_json_structure, 
    validate_api_key,
    create_query,
    extract_from_url,
    get_availability_status,
    DataQuery,
    TableData,
    ExtractionResult
)

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Data Extractor Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .success-box {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        .error-box {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
            margin: 1rem 0;
        }
        .table-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== SESSION STATE MANAGEMENT ====================

def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        'results': None,
        'current_query': '',
        'current_json': {},
        'current_raw_json': '',
        'history': [],
        'groq_api_key': '',
        'last_query_type': 'Simple Query'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_results():
    """Clear current session results"""
    st.session_state.results = None
    st.rerun()

def clear_all_session_data():
    """Clear all query-specific session data"""
    st.session_state.current_query = ''
    st.session_state.current_json = {}
    st.session_state.current_raw_json = ''
    st.session_state.results = None
    st.rerun()

# ==================== EXAMPLES AND CONSTANTS ====================

SIMPLE_EXAMPLES = [
    "unemployment rate United States",
    "population by state USA", 
    "electricity generation by source",
    "GDP by country",
    "aluminum consumption trends"
]

JSON_EXAMPLES = [
    {
        "Entity_Name": "Aluminium",
        "Entity_Type": "Commodity",
        "Region": "USA",
        "Metric": "Annual Consumption",
        "Unit": "Million Metric Tons"
    },
    {
        "Entity_Name": "Solar Energy",
        "Entity_Type": "Energy",
        "Region": "Global",
        "Metric": "Installed Capacity",
        "Unit": "Gigawatts"
    }
]

# ==================== QUERY INPUT INTERFACES ====================

def render_simple_query_interface() -> str:
    """Render simple text query interface"""
    st.subheader("🔍 Simple Query Search")
    
    current_value = st.session_state.get('current_query', '')
    
    query = st.text_area(
        "Enter your search query:",
        value=current_value,
        height=100,
        placeholder="e.g., 'aluminum consumption USA' or 'renewable energy statistics'",
        help="Enter any data-related search query",
        key="simple_query_input"
    )
    
    if query != st.session_state.get('current_query', ''):
        st.session_state.current_query = query
    
    # Example buttons
    st.write("**Quick Examples:**")
    cols = st.columns(2)
    for i, example in enumerate(SIMPLE_EXAMPLES[:4]):
        col = cols[i % 2]
        if col.button(f"📌 {example}", key=f"simple_ex_{i}"):
            st.session_state.current_query = example
            st.rerun()
    
    if st.button("🗑️ Clear Query", key="clear_simple"):
        st.session_state.current_query = ''
        st.rerun()
    
    return query.strip() if query else ''

def render_structured_json_interface() -> Optional[Dict[str, str]]:
    """Render structured JSON query interface"""
    st.subheader("📋 Structured JSON Query")
    
    preset = st.session_state.get('current_json', {})
    
    with st.form("structured_json_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            entity_name = st.text_input(
                "Entity Name *", 
                value=preset.get('Entity_Name', ''), 
                placeholder="e.g., Aluminum, Solar Energy"
            )
            entity_type = st.selectbox(
                "Entity Type", 
                ["", "Commodity", "Energy", "Technology", "Service", "Other"],
                index=0 if not preset.get('Entity_Type') else 
                      ["", "Commodity", "Energy", "Technology", "Service", "Other"].index(preset.get('Entity_Type', ''))
            )
            region = st.text_input(
                "Region", 
                value=preset.get('Region', ''), 
                placeholder="e.g., USA, China, Global"
            )
        
        with col2:
            metric = st.text_input(
                "Metric", 
                value=preset.get('Metric', ''), 
                placeholder="e.g., Annual Production, Consumption"
            )
            curve_type = st.selectbox(
                "Curve Type", 
                ["", "Adoption", "Production", "Consumption", "Cost", "Performance", "Other"],
                index=0 if not preset.get('Curve_Type') else 
                      ["", "Adoption", "Production", "Consumption", "Cost", "Performance", "Other"].index(preset.get('Curve_Type', ''))
            )
            unit = st.text_input(
                "Unit", 
                value=preset.get('Unit', ''), 
                placeholder="e.g., Million Tons, GWh"
            )
        
        form_submitted = st.form_submit_button("📝 Update Query", use_container_width=True)
    
    # Build query JSON
    query_json = {}
    if entity_name.strip():
        query_json = {
            "Entity_Name": entity_name.strip(),
            "Entity_Type": entity_type if entity_type else "",
            "Curve_Type": curve_type if curve_type else "",
            "Region": region.strip(),
            "Metric": metric.strip(),
            "Unit": unit.strip()
        }
        query_json = {k: v for k, v in query_json.items() if v}
    
    if form_submitted or query_json != st.session_state.get('current_json', {}):
        st.session_state.current_json = query_json
    
    # Show current query
    if query_json:
        with st.expander("📄 Current JSON Query", expanded=True):
            st.json(query_json)
            
            is_valid, error_msg = validate_json_structure(query_json)
            if is_valid:
                st.success("✅ Valid query structure")
            else:
                st.error(f"❌ {error_msg}")
    
    # Example buttons
    st.write("**Load Examples:**")
    cols = st.columns(len(JSON_EXAMPLES))
    for i, example in enumerate(JSON_EXAMPLES):
        example_name = f"{example['Entity_Name']}"
        if cols[i].button(f"📌 {example_name}", key=f"json_ex_{i}"):
            st.session_state.current_json = example.copy()
            st.rerun()
    
    if st.button("🗑️ Clear Form", key="clear_structured"):
        st.session_state.current_json = {}
        st.rerun()
    
    return query_json if query_json else None

def render_raw_json_interface() -> Optional[Dict[str, str]]:
    """Render raw JSON input interface"""
    st.subheader("📝 Raw JSON Input")
    
    with st.expander("📖 Expected JSON Format", expanded=False):
        example = {
            "Entity_Name": "Steel",
            "Entity_Type": "Commodity", 
            "Region": "China",
            "Metric": "Production Volume",
            "Unit": "Million Tons"
        }
        st.code(json.dumps(example, indent=2), language='json')
        st.caption("Required: Entity_Name. Optional: All other fields.")
    
    current_raw = st.session_state.get('current_raw_json', '')
    
    raw_json = st.text_area(
        "JSON Query:",
        value=current_raw,
        height=200,
        placeholder='{"Entity_Name": "Steel", "Region": "China", "Metric": "Production"}',
        help="Paste your JSON query here",
        key="raw_json_input"
    )
    
    if raw_json != st.session_state.get('current_raw_json', ''):
        st.session_state.current_raw_json = raw_json
    
    parsed_json = None
    validation_error = None
    
    if raw_json.strip():
        try:
            parsed_json = json.loads(raw_json)
            
            is_valid, error_msg = validate_json_structure(parsed_json)
            if is_valid:
                st.success("✅ Valid JSON and structure")
                with st.expander("🔍 Parsed Preview", expanded=False):
                    st.json(parsed_json)
            else:
                validation_error = error_msg
                st.error(f"❌ Structure Error: {error_msg}")
                st.info("💡 The JSON is valid but doesn't match expected structure")
                
        except json.JSONDecodeError as e:
            validation_error = f"Invalid JSON: {str(e)}"
            st.error(f"❌ {validation_error}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Format JSON", disabled=parsed_json is None):
            if parsed_json:
                formatted = json.dumps(parsed_json, indent=2)
                st.session_state.current_raw_json = formatted
                st.rerun()
    
    with col2:
        if st.button("✅ Validate", disabled=not raw_json.strip()):
            pass  # Validation happens automatically above
    
    with col3:
        if st.button("🗑️ Clear JSON"):
            st.session_state.current_raw_json = ''
            st.rerun()
    
    # Example buttons
    st.write("**Load JSON Examples:**")
    cols = st.columns(len(JSON_EXAMPLES))
    for i, example in enumerate(JSON_EXAMPLES):
        example_name = f"{example['Entity_Name']}"
        if cols[i].button(f"📌 {example_name}", key=f"raw_ex_{i}"):
            formatted_example = json.dumps(example, indent=2)
            st.session_state.current_raw_json = formatted_example
            st.rerun()
    
    return parsed_json if parsed_json and not validation_error else None

def render_direct_url_interface(enable_deep_crawl: bool):
    """Render direct URL extraction interface"""
    st.header("🌐 Direct URL Data Extraction")
    st.info("Extract tabular data directly from any URL with deep crawling capabilities")
    
    url_input = st.text_input(
        "Enter URL to extract data from:",
        placeholder="https://ourworldindata.org/grapher/annual-co2-emissions-per-country",
        help="Enter any URL that might contain tabular data"
    )
    
    # Sample URLs
    st.markdown("**Sample URLs to try:**")
    sample_urls = [
        ("Our World in Data - CO2 Emissions", "https://ourworldindata.org/grapher/annual-co2-coal?tab=line&time=earliest..2023&country=~CHN&tableSearch=china"),
        ("Our World in Data - Solar Capacity", "https://ourworldindata.org/grapher/solar-pv-cumulative-capacity"),
        ("Our World in Data - EV Sales", "https://ourworldindata.org/grapher/electric-car-sales")
    ]
    
    for i, (name, url) in enumerate(sample_urls):
        if st.button(f"📌 {name}", key=f"url_sample_{i}", use_container_width=True):
            st.session_state.sample_url = url
            st.rerun()
    
    # Use sample URL if selected
    if 'sample_url' in st.session_state:
        url_input = st.session_state.sample_url
        del st.session_state.sample_url
    
    if st.button("📊 Extract Data from URL", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL")
            return
        
        # Validate URL format
        from urllib.parse import urlparse
        try:
            parsed_url = urlparse(url_input)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                st.error("Invalid URL format")
                return
        except Exception:
            st.error("Invalid URL format")
            return
        
        with st.spinner(f"🔍 Extracting data from {url_input}..."):
            url_info = {
                'title': 'Direct URL',
                'confidence': 1.0,
                'found_via': 'direct_input'
            }
            
            tables = asyncio.run(extract_from_url(url_input, url_info, enable_deep_crawl))
            
            if tables:
                st.markdown("---")
                render_url_extraction_results(tables, url_input)
            else:
                st.error("❌ No tables found in the URL")
                st.info("Try enabling deep crawling or check if the URL contains tabular data")

# ==================== RESULT DISPLAY FUNCTIONS ====================

def render_url_extraction_results(tables: List[TableData], url: str):
    """Render results from direct URL extraction"""
    st.subheader("📊 Extraction Results")
    
    st.success(f"🎉 Successfully extracted {len(tables)} table(s)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tables Found", len(tables))
    with col2:
        total_rows = sum(t.rows for t in tables)
        st.metric("Total Rows", total_rows)
    with col3:
        total_cols = sum(t.cols for t in tables)
        st.metric("Total Columns", total_cols)
    
    # Display tables
    render_tables_display(tables, f"from {url}")

def render_tables_display(tables: List[TableData], source_description: str = ""):
    """Display extracted tables with visualizations"""
    if not tables:
        st.warning("No tables found")
        return
    
    st.success(f"Found {len(tables)} table(s) {source_description}")
    
    for i, table in enumerate(tables, 1):
        with st.expander(f"📊 Table {i}: {table.type} ({table.rows} × {table.cols})", expanded=True):
            
            # Table metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Type:** {table.type}")
            with col2:
                st.info(f"**Size:** {table.rows} × {table.cols}")
            with col3:
                st.info(f"**Confidence:** {table.confidence_score:.2f}")
            
            if table.source_url:
                st.write(f"**Source:** {table.source_url}")
            
            # Display table data
            if table.headers and table.data:
                try:
                    df = pd.DataFrame(table.data, columns=table.headers)
                    st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 100))
                    
                    # Auto-generate visualization
                    create_auto_visualization(df, table)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"extracted_table_{i}_{timestamp}.csv"
                    
                    st.download_button(
                        label=f"📥 Download Table {i} as CSV",
                        data=csv,
                        file_name=filename,
                        mime='text/csv',
                        key=f"download_table_{i}_{id(table)}",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error displaying table: {e}")
                    render_raw_table_fallback(table)

def render_raw_table_fallback(table: TableData):
    """Fallback display for problematic tables"""
    with st.expander("Raw Data", expanded=False):
        st.write("**Headers:**", table.headers)
        for j, row in enumerate(table.data[:5], 1):
            st.write(f"**Row {j}:**", row)
        if len(table.data) > 5:
            st.write(f"... and {len(table.data) - 5} more rows")

def create_auto_visualization(df: pd.DataFrame, table_info: TableData):
    """Auto-generate appropriate visualization for data"""
    try:
        time_col = None
        value_cols = []
        
        # Look for year/date columns
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['year', 'date', 'time']):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        time_col = col
                        break
                except:
                    continue
        
        # Find numeric value columns
        for col in df.columns:
            if col != time_col:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all() and df[col].dtype in ['int64', 'float64']:
                        value_cols.append(col)
                except:
                    continue
        
        # Create appropriate visualization
        if time_col and value_cols and len(df) > 1:
            chart_df = df.dropna(subset=[time_col] + value_cols[:1])
            
            if len(chart_df) > 1:
                value_col = value_cols[0]
                
                if 'year' in time_col.lower():
                    chart_df = chart_df[
                        (chart_df[time_col] >= 1900) & 
                        (chart_df[time_col] <= 2030)
                    ]
                
                if len(chart_df) > 1:
                    fig = px.line(
                        chart_df, 
                        x=time_col, 
                        y=value_col,
                        title=f"{value_col} over {time_col}",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
        elif len(value_cols) >= 2:
            chart_df = df.dropna(subset=value_cols[:2])
            if len(chart_df) > 1:
                fig = px.scatter(
                    chart_df, 
                    x=value_cols[0], 
                    y=value_cols[1],
                    title=f"{value_cols[1]} vs {value_cols[0]}"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception:
        pass  # Silently fail for visualization errors

def render_extraction_results(result: ExtractionResult):
    """Render main extraction results"""
    if not result.success:
        st.error(f"❌ {result.error or 'Unknown error'}")
        
        # Offer LLM fallback for failed extractions
        if not result.used_llm_fallback:
            st.divider()
            st.subheader("🤖 Try AI Data Generation")
            st.info("Generate data using AI when web sources are insufficient.")
            
            if st.button("🤖 Generate AI Data", type="primary"):
                if result.query and st.session_state.get('groq_api_key'):
                    ai_result = run_extraction_with_progress(result.query, st.session_state.groq_api_key, 0, False)
                    if ai_result:
                        st.session_state.results = ai_result
                        st.rerun()
        return
    
    st.divider()
    st.header("📊 Extraction Results")
    
    # Query info
    query_info = result.query
    if isinstance(query_info, dict):
        st.info(f"**Query:** {query_info.get('Entity_Name', 'Unknown')} - {query_info.get('Metric', 'Unknown')}")
    else:
        st.info(f"**Query:** {str(query_info)[:100]}...")
    
    # LLM fallback indicator
    if result.used_llm_fallback:
        st.success("🤖 **AI Enhancement**: Generated data when web sources were insufficient")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("URLs Found", result.summary.get('urls_discovered', 0))
    with col2:
        st.metric("Successful", result.summary.get('successful_urls', 0))
    with col3:
        st.metric("Tables", result.summary.get('total_tables', 0))
    with col4:
        st.metric("Data Rows", result.summary.get('total_rows', 0))
    
    # Sources section
    render_sources_section(result)
    
    # Tables section
    render_tables_section(result)
    
    # AI Analysis
    # render_analysis_section(result)
    
    # Downloads
    render_downloads_section(result)

def render_sources_section(result: ExtractionResult):
    """Render discovered sources section"""
    st.subheader("🔍 Data Sources")
    
    for i, url_info in enumerate(result.discovered_urls, 1):
        url = url_info['url']
        url_result = result.url_results.get(url, {})
        
        if url == 'LLM_FALLBACK':
            st.info(f"🤖 **{i}. AI Generated Data**\nConfidence: {url_info['confidence']:.2f}")
        else:
            success = url_result.get('success', False)
            icon = "✅" if success else "❌"
            
            with st.expander(f"{icon} {i}. {url_info['title']}", expanded=False):
                st.write(f"**URL:** {url}")
                st.write(f"**Confidence:** {url_info['confidence']:.2f}")
                
                if success:
                    tables_found = url_result.get('tables_found', 0)
                    st.write(f"**Tables found:** {tables_found}")
                else:
                    st.error(f"**Error:** {url_result.get('error', 'Unknown')}")

def render_tables_section(result: ExtractionResult):
    """Render extracted tables section"""
    tables = result.tables
    if not tables:
        return
    
    st.subheader("📋 Extracted Data")
    
    # Table selection for multiple tables
    selected_table = tables[0]
    
    if len(tables) > 1:
        table_options = []
        for table in tables:
            source_type = "🤖 AI" if table.get('found_via') == 'llm_fallback' else "🌐 Web"
            if table.get('found_via') == 'file_extraction':
                source_type = "📁 File"
            
            title = table.get('source_title', 'Unknown')[:35]
            option = f"{source_type} | {table['id']}: {title} ({table['rows']}×{table['cols']})"
            table_options.append(option)
        
        selected_idx = st.selectbox("Select table:", range(len(table_options)), 
                                  format_func=lambda x: table_options[x])
        selected_table = tables[selected_idx]
    
    # Display selected table
    render_single_table_display(selected_table)

def render_single_table_display(table: Dict[str, Any]):
    """Display individual table with metadata"""
    # Table metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        table_type = get_display_type(table.get('type', 'unknown'))
        st.info(f"**Type:** {table_type}")
    
    with col2:
        st.info(f"**Size:** {table['rows']} × {table['cols']}")
    
    with col3:
        confidence = table.get('confidence_score', 0)
        st.info(f"**Confidence:** {confidence:.2f}")
    
    # Source information
    source_title = table.get('source_title', 'Unknown')
    source_url = table.get('source_url', 'Unknown')
    
    if source_url == 'LLM_FALLBACK':
        st.warning(f"**Source:** {source_title}")
        st.caption("⚠️ AI-generated data. Verify for current figures.")
    else:
        st.write(f"**Source:** {source_title}")
        if source_url != 'Unknown':
            st.write(f"**URL:** {source_url}")
    
    # Display table data
    if table.get('data'):
        try:
            df = pd.DataFrame(table['data'], columns=table['headers'])
            st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 100))
            st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")
            
            # Fallback: show raw data
            with st.expander("Raw Data", expanded=False):
                st.write("**Headers:**", table['headers'])
                for i, row in enumerate(table['data'][:5], 1):
                    st.write(f"**Row {i}:**", row)
                if len(table['data']) > 5:
                    st.write(f"... and {len(table['data']) - 5} more rows")
    else:
        st.warning("No data in table")

def get_display_type(table_type: str) -> str:
    """Get display name for table type"""
    type_map = {
        'llm_generated': "🤖 AI Generated",
        'html_table': "🌐 Web Table",
        'csv_file': "📈 CSV File",
        'file_extraction': "📁 File Extract"
    }
    return type_map.get(table_type, "📋 Table")

def render_analysis_section(result: ExtractionResult):
    """Render AI analysis section"""
    analysis = result.response
    if not analysis:
        return
    
    st.subheader("🧠 AI Analysis")
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            padding: 1.5rem; 
            border-radius: 10px; 
            border-left: 4px solid #0066cc;
            line-height: 1.6;
            margin: 1rem 0;
        ">
            {analysis.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_downloads_section(result: ExtractionResult):
    """Render download options section"""
    st.subheader("📥 Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        json_data = {
            'query': result.query,
            'results': result._asdict(),
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            "📄 Download JSON",
            data=json.dumps(json_data, indent=2, default=str),
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV download
        if result.tables:
            try:
                table = result.tables[0]
                df = pd.DataFrame(table['data'], columns=table['headers'])
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception:
                st.button("📊 CSV (Error)", disabled=True, use_container_width=True)
        else:
            st.button("📊 No Tables", disabled=True, use_container_width=True)
    
    with col3:
        # Report download
        report = generate_text_report(result)
        st.download_button(
            "📋 Download Report",
            data=report,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_text_report(result: ExtractionResult) -> str:
    """Generate text report from extraction results"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    query_str = result.query
    if isinstance(query_str, dict):
        query_str = f"{query_str.get('Entity_Name', '')} {query_str.get('Metric', '')}".strip()
    
    report = f"""DATA EXTRACTION REPORT
=====================

Generated: {timestamp}
Query: {query_str}
LLM Fallback Used: {'Yes' if result.used_llm_fallback else 'No'}

ANALYSIS:
{result.response or 'No analysis available'}

SUMMARY:
- URLs Discovered: {result.summary.get('urls_discovered', 0)}
- Successful Extractions: {result.summary.get('successful_urls', 0)}
- Total Tables: {result.summary.get('total_tables', 0)}
- Total Rows: {result.summary.get('total_rows', 0)}

SOURCES:
"""
    
    for i, url_info in enumerate(result.discovered_urls, 1):
        url = url_info['url']
        status = result.url_results.get(url, {})
        success = "SUCCESS" if status.get('success') else "FAILED"
        
        if url == 'LLM_FALLBACK':
            report += f"{i}. AI GENERATED DATA\n"
        else:
            report += f"{i}. {success} - {url_info['title']}\n"
            report += f"   URL: {url}\n"
    
    return report

# ==================== EXTRACTION RUNNER ====================

def run_extraction_with_progress(query: Union[str, dict], api_key: str, max_urls: int, enable_crawl: bool) -> Optional[ExtractionResult]:
    """Run extraction with progress tracking"""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Validate inputs
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            progress.empty()
            status.error(f"❌ Validation Error: {error_msg}")
            return None
        
        status.text("🔎 Searching for data sources...")
        progress.progress(25)
        time.sleep(0.5)
        
        status.text("📊 Extracting tables from sources...")
        progress.progress(50)
        
        # Run actual extraction
        result = asyncio.run(extract_data(query, api_key, max_urls, enable_crawl))
        
        if result.used_llm_fallback:
            status.text("🤖 Applied AI fallback generation...")
            progress.progress(75)
            time.sleep(0.3)
        
        status.text("🧠 Analyzing extracted data...")
        progress.progress(90)
        time.sleep(0.3)
        
        progress.progress(100)
        status.text("✅ Extraction complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress.empty()
        status.empty()
        
        return result
        
    except Exception as e:
        progress.empty()
        status.error(f"❌ Error: {str(e)}")
        st.error(f"Extraction failed: {str(e)}")
        return None

# ==================== SIDEBAR ====================

def render_sidebar():
    """Render sidebar with settings and status"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Required for LLM analysis and fallback generation",
            value=st.session_state.get('groq_api_key', '')
        )
        
        if groq_api_key:
            if validate_api_key(groq_api_key):
                st.success("✅ API Key Valid")
                st.session_state.groq_api_key = groq_api_key
            else:
                st.error("❌ Invalid API Key Format")
                st.session_state.groq_api_key = None
        else:
            st.warning("⚠️ API Key Required")
            st.session_state.groq_api_key = None
        
        st.divider()
        
        # Extraction settings
        max_urls = st.slider("Max URLs to Check", 1, 10, 5)
        enable_deep_crawl = st.checkbox("Deep File Crawling", value=True, 
                                      help="Extract data from linked PDF, Excel, CSV files")
        
        st.divider()
        
        # Status
        st.subheader("📊 Status")
        availability = get_availability_status()
        st.write(f"**crawl4ai:** {'✅ Available' if availability['crawl4ai'] else '❌ Not installed'}")
        st.write(f"**Groq:** {'✅ Available' if availability['groq'] else '❌ Not installed'}")
        
        # Debug info
        if st.checkbox("Show Debug Info"):
            st.subheader("🐛 Debug")
            st.write(f"Current Query: {st.session_state.get('current_query', 'None')[:50]}...")
            st.write(f"Current JSON: {bool(st.session_state.get('current_json', {}))}")
            st.write(f"Raw JSON: {bool(st.session_state.get('current_raw_json', '').strip())}")
            st.write(f"API Key: {'Set' if st.session_state.get('groq_api_key') else 'Not Set'}")
    
    return groq_api_key, max_urls, enable_deep_crawl

# ==================== MAIN APPLICATION ====================

def main():
    """Main Streamlit application"""
    apply_custom_css()
    
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Data Extractor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Render sidebar and get settings
    groq_api_key, max_urls, enable_deep_crawl = render_sidebar()
    
    # Query type selection
    query_type = st.radio(
        "**Query Type:**",
        ["Simple Query", "Structured JSON", "Raw JSON Input", "Direct URL"],
        horizontal=True,
        help="Choose how you want to input your query"
    )
    
    st.session_state.last_query_type = query_type
    
    # Query interfaces
    query = None
    
    if query_type == "Simple Query":
        query = render_simple_query_interface()
    elif query_type == "Structured JSON":
        query = render_structured_json_interface()
    elif query_type == "Raw JSON Input":
        query = render_raw_json_interface()
    elif query_type == "Direct URL":
        render_direct_url_interface(enable_deep_crawl)
        return  # Exit early for direct URL tab
    
    # Validation status
    if query:
        is_valid, error_msg = validate_query(query)
        if is_valid:
            st.success(f"✅ Query ready: {type(query).__name__}")
        else:
            st.error(f"❌ Query issue: {error_msg}")
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Check if extraction can run
        has_api_key = st.session_state.get('groq_api_key') is not None
        has_valid_query = query is not None and validate_query(query)[0]
        can_extract = has_api_key and has_valid_query
        
        # Show reason if can't extract
        if not can_extract:
            reasons = []
            if not has_api_key:
                reasons.append("API Key required")
            if not has_valid_query:
                reasons.append("Valid query required")
            
            help_text = f"Missing: {', '.join(reasons)}"
        else:
            help_text = "Ready to extract data"
        
        extract_button = st.button(
            "🚀 Extract Data", 
            type="primary", 
            disabled=not can_extract,
            use_container_width=True,
            help=help_text
        )
    
    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            clear_session_results()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_all_session_data()
    
    # Handle extraction
    if extract_button:
        if not st.session_state.get('groq_api_key'):
            st.error("❌ Please provide a valid Groq API key")
        elif not query:
            st.error("❌ Please enter a valid query")
        else:
            st.info(f"🔄 Extracting data for {query_type}: {type(query).__name__}")
            
            # Run extraction
            result = run_extraction_with_progress(query, st.session_state.groq_api_key, max_urls, enable_deep_crawl)
            
            if result:
                st.session_state.results = result
                
                # Add to history
                st.session_state.history.append(result)
                if len(st.session_state.history) > 10:  # Keep last 10
                    st.session_state.history = st.session_state.history[-10:]
                
                st.rerun()
    
    # Display results
    if st.session_state.results:
        render_extraction_results(st.session_state.results)

if __name__ == "__main__":
    main()