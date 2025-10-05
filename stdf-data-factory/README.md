# 🔍 Data Hunting Agent

A unified intelligent data extraction and analysis platform that combines PDF processing, chart analysis, and web data extraction into a single workflow-based application.

## ✨ Features

### 🎯 Workflow-Based Interface
- **Quick Start Landing Page**: Guided workflow selection based on your data source
- **Dynamic Content Area**: Interface adapts to your chosen workflow
- **Clean Navigation**: Easy back-and-forth between workflows

### 📊 Multiple Data Sources
1. **📄 PDF Analysis**: Extract time series data from PDF documents containing charts
2. **📈 Chart Image Analysis**: Analyze individual chart images for numerical data
3. **🌐 Web Data Extraction**: Search and extract structured data from web sources
4. **🔗 URL Extraction**: Direct extraction from specific URLs with deep crawling

### 🤖 AI-Powered Analysis
- **Advanced Chart Recognition**: Using Grok (XAI) for image analysis
- **Intelligent Data Extraction**: Using Groq for structured data processing
- **Smart Query Understanding**: Natural language and structured queries
- **Automatic Visualization**: Interactive charts generated from extracted data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys (see Environment Setup)

### Environment Setup
Set these environment variables before running the app:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export XAI_API_KEY="your_xai_api_key_here"
```

### Running the App

#### Option 1: Using the Runner Script (Recommended)
```bash
python3 run_unified.py
```

#### Option 2: Direct Streamlit Command
```bash
streamlit run unified_app.py
```

The app will be available at `http://localhost:8501`

## 📋 Workflows Guide

### 1. 📄 PDF Analysis
**Use when**: You have PDF documents with charts, graphs, or time series visualizations

**Process**:
1. Upload your PDF file
2. Click "Analyze PDF"
3. View extracted charts and download structured data

**Output**: Interactive visualizations, structured JSON data, downloadable CSV

### 2. 📊 Chart Image Analysis
**Use when**: You have individual chart or graph images

**Process**:
1. Upload image file (PNG, JPG, JPEG)
2. Click "Analyze Chart" 
3. View extracted numerical data

**Output**: Structured chart data, interactive plots, JSON export

### 3. 🌐 Web Data Extraction
**Use when**: You need to find and extract data from web sources

**Query Types**:
- **Simple Search**: Natural language queries (e.g., "aluminum consumption USA")
- **Structured Query**: Form-based query builder
- **JSON Query**: Advanced structured queries

**Process**:
1. Choose query type
2. Enter your search parameters
3. Click "Extract Data"
4. Review discovered sources and extracted tables

### 4. 🔗 URL Extraction
**Use when**: You have a specific URL containing tabular data

**Process**:
1. Enter the URL
2. Click "Extract Data from URL"
3. View extracted tables and data

**Supported Sources**: Our World in Data, Wikipedia tables, CSV files, Excel files, and more

## 🔧 Configuration

The app uses sensible defaults with no frontend configuration needed:

- **Max URLs**: 5 (for web extraction)
- **Deep Crawling**: Enabled (extracts from linked files)
- **API Keys**: From environment variables only
- **Processing**: Optimized settings for best results

## 📁 File Structure

```
stdf-data-factory/
├── unified_app.py          # Main unified application
├── run_unified.py          # Runner script
├── app.py                  # Original PDF/Chart analyzer
├── main.py                 # Original data extractor
├── data_extractor.py       # Core extraction logic
├── utils/                  # Utility modules
│   ├── api_clients.py      # API client implementations
│   ├── image_processor.py  # Image processing utilities
│   └── pdf_processor.py    # PDF processing utilities
├── uploads/                # Uploaded files
├── extracted_data/         # Analysis results
└── requirements.txt        # Python dependencies
```

## 🎨 User Interface

### Quick Start Page
- **Clean Landing**: Choose your workflow based on data source
- **Feature Overview**: Understand capabilities at a glance
- **Guided Selection**: Reduces decision paralysis

### Workflow Pages
- **Back Navigation**: Easy return to Quick Start
- **Contextual Interfaces**: Optimized for each workflow type
- **Progress Indicators**: Clear feedback during processing
- **Results Integration**: Seamless display of analysis results

### Results Display
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Data Tables**: Sortable, filterable data display
- **Download Options**: JSON, CSV, and report formats
- **Analysis Summary**: Key metrics and insights

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **OpenAI**: For Grok (XAI) API integration
- **Groq**: For LLM-based data analysis
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **crawl4ai**: Web crawling and extraction

### API Integration
- **Grok (XAI)**: Chart image analysis and data extraction
- **Groq**: Natural language processing and structured data generation
- **Environment-based**: All API keys from environment variables

### Data Processing
- **PDF Rendering**: High-resolution page extraction
- **Image Analysis**: AI-powered chart recognition
- **Web Crawling**: Intelligent table detection and extraction
- **Data Validation**: Automatic quality checks and cleaning

## 🔍 Example Use Cases

### Research & Analysis
- Extract economic data from government reports
- Analyze charts from academic papers
- Gather industry statistics from web sources

### Business Intelligence
- Process financial reports and charts
- Extract market data from online sources
- Analyze competitor data from public sources

### Academic Research
- Extract data from research papers
- Analyze historical trends from various sources
- Compile datasets from multiple web sources

## 🚨 Troubleshooting

### Common Issues

**"Environment variables not set"**
- Ensure GROQ_API_KEY and XAI_API_KEY are exported
- Restart terminal after setting variables

**"No charts detected"**
- Ensure image is clear and contains visible charts
- Try higher resolution images
- Check that charts have clear axes and labels

**"No tables found in URL"**
- Verify URL contains tabular data
- Try enabling deep crawling
- Check if site requires authentication

### Getting Help
1. Check environment variable setup
2. Verify API key validity
3. Ensure all dependencies are installed
4. Review error messages in the UI

## 📝 License

This project combines multiple data extraction and analysis tools into a unified platform for research and analysis purposes.

---

## 🎯 Next Steps

After running the app:
1. Visit `http://localhost:8501`
2. Choose your workflow from the Quick Start page
3. Follow the guided interface for your data source
4. Explore the interactive results and download options

**Happy analyzing!** 🚀
