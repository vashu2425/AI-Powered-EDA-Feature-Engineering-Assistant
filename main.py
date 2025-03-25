"""
AI-Powered EDA & Feature Engineering Assistant

This application enables users to upload a CSV dataset, and utilizes LLMs to analyze
the dataset to provide EDA and feature engineering recommendations.
"""

import streamlit as st
import pandas as pd
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import time
import logging

# Import local modules
from eda_analysis import DatasetAnalyzer
from llm_inference import LLMInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI-Powered EDA & Feature Engineering Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize our classes
@st.cache_resource
def get_llm_inference():
    try:
        return LLMInference()
    except Exception as e:
        st.error(f"Error initializing LLM inference: {str(e)}")
        return None

llm_inference = get_llm_inference()

# Session state initialization
if "dataset_analyzer" not in st.session_state:
    st.session_state.dataset_analyzer = DatasetAnalyzer()

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = {}

if "visualizations" not in st.session_state:
    st.session_state.visualizations = {}

if "eda_insights" not in st.session_state:
    st.session_state.eda_insights = ""

if "feature_engineering_recommendations" not in st.session_state:
    st.session_state.feature_engineering_recommendations = ""

if "data_quality_insights" not in st.session_state:
    st.session_state.data_quality_insights = ""

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "welcome"

# Custom CSS for better styling - dark mode theme
st.markdown("""
<style>
    /* Color palette - Dark Mode */
    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --secondary: #06B6D4;
        --secondary-light: #22D3EE;
        --accent: #FB7185;
        --bg-dark: #111827;
        --bg-card: #1F2937;
        --text-light: #F9FAFB;
        --text-muted: #9CA3AF;
        --text-dark: #D1D5DB;
        --border-dark: #374151;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --shadow: rgba(0, 0, 0, 0.3);
    }

    /* Base styles */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-light);
    }
    
    /* Override Streamlit defaults */
    .st-emotion-cache-16idsys p, .st-emotion-cache-16idsys div {
        color: var(--text-light);
    }
    
    .st-emotion-cache-16idsys h1, 
    .st-emotion-cache-16idsys h2, 
    .st-emotion-cache-16idsys h3, 
    .st-emotion-cache-16idsys h4 {
        color: var(--text-light);
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-light);
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
        padding-left: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-light);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        position: relative;
    }
    
    .section-header:after {
        content: "";
        position: absolute;
        width: 100%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, var(--secondary) 0%, var(--primary-light) 100%);
        border-radius: 3px;
    }

    /* Card styles */
    .card {
        background-color: var(--bg-card);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 15px var(--shadow);
        margin-bottom: 24px;
        border-top: 4px solid var(--primary);
        transition: transform 0.2s, box-shadow 0.2s;
        color: var(--text-light);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px var(--shadow);
    }
    
    /* Insight box styles */
    .insight-box {
        border-left: 4px solid var(--primary);
        padding: 15px;
        background-color: var(--bg-card);
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px var(--shadow);
        transition: transform 0.2s;
        color: var(--text-light);
    }
    
    .insight-box:hover {
        transform: translateX(5px);
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--border-dark);
        padding-bottom: 10px;
    }
    
    .nav-tab {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid transparent;
        color: var(--text-light);
    }
    
    .nav-tab:hover {
        color: var(--primary-light);
        background-color: rgba(99, 102, 241, 0.1);
    }
    
    .nav-tab.active {
        color: var(--primary-light);
        border-bottom: 3px solid var(--primary);
        background-color: var(--bg-card);
        border-top: 1px solid var(--border-dark);
        border-left: 1px solid var(--border-dark);
        border-right: 1px solid var(--border-dark);
        border-bottom: none;
    }
    
    /* Button styles */
    .custom-button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
        text-decoration: none;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.25);
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(99, 102, 241, 0.35);
    }
    
    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--error);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Chat UI elements */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .chat-input {
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }

    .chat-message {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 12px;
        box-shadow: 0 2px 5px var(--shadow);
    }
    
    .user-message {
        align-self: flex-end;
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border-bottom-right-radius: 2px;
    }
    
    .ai-message {
        align-self: flex-start;
        background-color: var(--bg-card);
        border-bottom-left-radius: 2px;
        border-left: 3px solid var(--secondary);
        color: var(--text-light);
    }
    
    /* Sidebar customization */
    .sidebar .sidebar-content {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-dark);
    }
    
    /* Dataframe styling */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px var(--shadow);
        margin-bottom: 20px;
        background-color: var(--bg-card);
    }
    
    /* Visual indicator for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--bg-card);
        border-radius: 10px 10px 0 0;
        padding: 5px 5px 0 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--border-dark);
        border-bottom: none;
        background-color: var(--bg-dark);
        font-weight: 600;
        color: var(--text-light);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        border-top: 3px solid var(--primary) !important;
        color: var(--primary-light) !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--bg-card);
        border-radius: 0 0 10px 10px;
        padding: 20px;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: var(--primary-light);
    }
    
    /* File uploader */
    .stFileUploader label {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
    }
    
    /* Override DataFrame style */
    .stDataFrame {
        background-color: var(--bg-card);
    }
    
    .stDataFrame td, .stDataFrame th {
        color: var(--text-light) !important;
    }
    
    .stDataFrame [data-testid="stVerticalBlock"] div:has(table) {
        background-color: var(--bg-card);
    }
    
    /* Override selectbox and other inputs */
    .stSelectbox label, .stTextInput label {
        color: var(--text-light) !important;
    }
    
    .stTextInput input, .stSelectbox select {
        background-color: var(--bg-card) !important;
        color: var(--text-light) !important;
        border-color: var(--border-dark) !important;
    }
    
    /* Override code blocks */
    code {
        background-color: #2D3748 !important;
        color: #E5E7EB !important;
    }
    
    /* Fix button text colors */
    button[kind="secondary"] {
        color: var(--text-dark) !important;
    }
    
    /* Fix download button */
    .stDownloadButton button {
        background-color: var(--primary) !important;
        color: white !important;
    }
    
    /* Fix text in messages */
    .stAlert p {
        color: currentColor !important;
    }
    
    /* Fix image captions */
    .stImage img + div {
        color: var(--text-muted) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def convert_df_to_csv_download_link(df):
    """Convert a DataFrame to a CSV download link"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def convert_text_to_download_link(text, filename="report.txt"):
    """Convert a text to a download link"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'data:file/txt;base64,{b64}'
    return href

def display_image_from_base64(base64_string, caption=""):
    """Display an image from a base64 string"""
    if base64_string:
        st.image(f"data:image/png;base64,{base64_string}", caption=caption, use_column_width=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<p class="main-header">AI-Powered EDA & Feature Engineering Assistant</p>', unsafe_allow_html=True)
    
    if llm_inference is None:
        st.error("Error: Could not initialize the LLM. Please check your Hugging Face token in the .env file.")
        st.stop()

    # Sidebar with dataset uploader & options
    with st.sidebar:
        st.markdown('<p class="section-header">Dataset Upload</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                
                # Create a download link for the original dataset
                st.markdown('<p class="section-header">Dataset Actions</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<a href="{convert_df_to_csv_download_link(df)}" class="custom-button" download="original_dataset.csv">'
                    f'📥 Download CSV</a>', 
                    unsafe_allow_html=True
                )
                
                if not st.session_state.dataset_loaded or st.button("🔄 Reload Analysis"):
                    with st.spinner("Analyzing dataset..."):
                        # Process the dataset
                        dataset_analyzer = DatasetAnalyzer(df)
                        st.session_state.dataset_analyzer = dataset_analyzer
                        dataset_info = dataset_analyzer.analyze_dataset()
                        st.session_state.dataset_info = dataset_info
                        
                        # Generate visualizations
                        visualizations = dataset_analyzer.generate_eda_visualizations()
                        st.session_state.visualizations = visualizations
                        
                        # Mark as loaded and set active tab to overview
                        st.session_state.dataset_loaded = True
                        st.session_state.active_tab = "overview"
                        
                        st.success("Dataset loaded and analyzed successfully!")
                
                # Show dataset information
                st.markdown('<p class="section-header">Dataset Summary</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-box">Rows: {df.shape[0]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-box">Columns: {df.shape[1]}</div>', unsafe_allow_html=True)
                
                # Quick access to data processing suggestions
                if st.session_state.dataset_loaded:
                    missing_count = sum(1 for _, (count, _) in st.session_state.dataset_info["missing_values"].items() if count > 0)
                    if missing_count > 0:
                        st.markdown(f'<div class="badge badge-warning">⚠️ {missing_count} columns with missing values</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please make sure you're uploading a valid CSV file.")
    
    # Main content area - Single set of tabs
    if st.session_state.dataset_loaded:
        # Create a single navigation system
        st.markdown('<div class="nav-tabs">', unsafe_allow_html=True)
        
        # Create colorful navigation tabs
        tab_styles = {
            "overview": "background: linear-gradient(90deg, #4338CA, #6366F1); color: white;",
            "eda": "background: linear-gradient(90deg, #0891B2, #06B6D4); color: white;",
            "feature_engineering": "background: linear-gradient(90deg, #7C3AED, #8B5CF6); color: white;",
            "data_quality": "background: linear-gradient(90deg, #D97706, #F59E0B); color: white;",
            "chat": "background: linear-gradient(90deg, #059669, #10B981); color: white;"
        }
        
        # Generate additional styling for active tabs
        overview_style = tab_styles["overview"] + (" box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "overview" else "")
        eda_style = tab_styles["eda"] + (" box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "eda" else "")
        fe_style = tab_styles["feature_engineering"] + (" box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "feature_engineering" else "")
        dq_style = tab_styles["data_quality"] + (" box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "data_quality" else "")
        chat_style = tab_styles["chat"] + (" box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "chat" else "")
        
        # Create large, colorful tabs with better clickable area
        cols = st.columns(5)
        
        # Define tabs with custom styling
        with cols[0]:
            if st.button("📊 Dataset Overview", key="btn_overview", 
                       help="View dataset overview, statistics and visualizations",
                       use_container_width=True):
                st.session_state.active_tab = "overview"
                st.experimental_rerun()
                
        with cols[1]:
            if st.button("🔍 EDA Insights", key="btn_eda",
                       help="Get AI-generated exploratory data analysis",
                       use_container_width=True):
                st.session_state.active_tab = "eda"
                st.experimental_rerun()
                
        with cols[2]:
            if st.button("🔧 Feature Engineering", key="btn_fe",
                       help="Get feature engineering recommendations",
                       use_container_width=True):
                st.session_state.active_tab = "feature_engineering"
                st.experimental_rerun()
                
        with cols[3]:
            if st.button("⚠️ Data Quality", key="btn_dq",
                       help="View data quality insights and issues",
                       use_container_width=True):
                st.session_state.active_tab = "data_quality"
                st.experimental_rerun()
                
        with cols[4]:
            if st.button("💬 Chat with Data", key="btn_chat",
                       help="Ask questions about your dataset",
                       use_container_width=True):
                st.session_state.active_tab = "chat"
                st.experimental_rerun()
        
        # Highlight the active tab with some CSS magic
        highlight_css = f"""
        <style>
            /* Override button styles with custom styling for tabs */
            div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {{
                background: {tab_styles["overview"] if st.session_state.active_tab == "overview" else "var(--bg-card)"};
                border-radius: 10px;
                padding: 10px 0;
                font-weight: bold;
                border: none;
                {("box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "overview" else "")}
            }}
            
            div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {{
                background: {tab_styles["eda"] if st.session_state.active_tab == "eda" else "var(--bg-card)"};
                border-radius: 10px;
                padding: 10px 0;
                font-weight: bold;
                border: none;
                {("box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "eda" else "")}
            }}
            
            div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {{
                background: {tab_styles["feature_engineering"] if st.session_state.active_tab == "feature_engineering" else "var(--bg-card)"};
                border-radius: 10px;
                padding: 10px 0;
                font-weight: bold;
                border: none;
                {("box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "feature_engineering" else "")}
            }}
            
            div[data-testid="stHorizontalBlock"] > div:nth-child(4) button {{
                background: {tab_styles["data_quality"] if st.session_state.active_tab == "data_quality" else "var(--bg-card)"};
                border-radius: 10px;
                padding: 10px 0;
                font-weight: bold;
                border: none;
                {("box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "data_quality" else "")}
            }}
            
            div[data-testid="stHorizontalBlock"] > div:nth-child(5) button {{
                background: {tab_styles["chat"] if st.session_state.active_tab == "chat" else "var(--bg-card)"};
                border-radius: 10px;
                padding: 10px 0;
                font-weight: bold;
                border: none;
                {("box-shadow: 0 8px 15px rgba(0,0,0,0.2); transform: translateY(-3px);" if st.session_state.active_tab == "chat" else "")}
            }}
            
            /* Make buttons look better */
            button[kind="secondary"] {{
                font-size: 1rem;
                padding: 0.5rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 80px;
            }}
            
            /* Add emoji size */
            button[kind="secondary"]::before {{
                content: attr(data-emoji);
                font-size: 1.5rem;
                margin-bottom: 0.25rem;
            }}
        </style>
        """
        
        st.markdown(highlight_css, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display content based on active tab
        if st.session_state.active_tab == "overview":
            st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
            
            # Display dataset sample in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df = st.session_state.dataset_analyzer.df
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
            
            # Create tabs for different statistics views
            stats_tab1, stats_tab2, stats_tab3 = st.tabs(["📈 Numerical Stats", "📊 Column Info", "🔢 Missing Values"])
            
            with stats_tab1:
                # Display numerical statistics
                numerical_columns = st.session_state.dataset_info["numerical_columns"]
                if numerical_columns:
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(df[numerical_columns].describe(), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No numerical columns found in the dataset.")
            
            with stats_tab2:
                # Display column information
                column_info = []
                for col in df.columns:
                    dtype = st.session_state.dataset_info["dtypes"].get(col, "unknown")
                    unique_count = st.session_state.dataset_info["unique_values"].get(col, 0)
                    missing_count, missing_percent = st.session_state.dataset_info["missing_values"].get(col, (0, 0))
                    
                    column_info.append({
                        "Column": col,
                        "Type": dtype,
                        "Unique Values": unique_count,
                        "Missing Values": missing_count,
                        "Missing %": f"{missing_percent}%"
                    })
                
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(column_info), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with stats_tab3:
                # Display missing values information
                missing_data = []
                for col, (count, percent) in st.session_state.dataset_info["missing_values"].items():
                    if count > 0:
                        missing_data.append({
                            "Column": col,
                            "Missing Count": count,
                            "Missing Percentage": f"{percent}%"
                        })
                
                if missing_data:
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Plot missing values
                    if "missing_values_heatmap" in st.session_state.visualizations:
                        st.markdown("### Missing Values Heatmap")
                        display_image_from_base64(
                            st.session_state.visualizations["missing_values_heatmap"],
                            "Missing values heatmap (yellow indicates missing values)"
                        )
                else:
                    st.markdown('<div class="badge badge-success">✅ No missing values found in the dataset!</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown('<p class="section-header">Visualizations</p>', unsafe_allow_html=True)
            
            # Show visualizations in a grid
            if st.session_state.visualizations:
                visualization_types = {
                    "Distributions": [k for k in st.session_state.visualizations.keys() if k.startswith("distribution_")],
                    "Categorical": [k for k in st.session_state.visualizations.keys() if k.startswith("categorical_")],
                    "Relationships": ["correlation_heatmap", "scatter_plot"],
                }
                
                viz_tabs = st.tabs(list(visualization_types.keys()))
                
                # Distributions Tab
                with viz_tabs[0]:
                    if visualization_types["Distributions"]:
                        for viz_key in visualization_types["Distributions"]:
                            column_name = viz_key.replace("distribution_", "")
                            st.markdown(f"### Distribution of {column_name}")
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            display_image_from_base64(st.session_state.visualizations[viz_key])
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No distribution plots available.")
                
                # Categorical Tab
                with viz_tabs[1]:
                    if visualization_types["Categorical"]:
                        for viz_key in visualization_types["Categorical"]:
                            column_name = viz_key.replace("categorical_", "")
                            st.markdown(f"### Distribution of {column_name}")
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            display_image_from_base64(st.session_state.visualizations[viz_key])
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No categorical plots available.")
                
                # Relationships Tab
                with viz_tabs[2]:
                    if "correlation_heatmap" in st.session_state.visualizations:
                        st.markdown("### Correlation Heatmap")
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        display_image_from_base64(st.session_state.visualizations["correlation_heatmap"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if "scatter_plot" in st.session_state.visualizations:
                        st.markdown("### Scatter Plot (Most Correlated Features)")
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        display_image_from_base64(st.session_state.visualizations["scatter_plot"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if "correlation_heatmap" not in st.session_state.visualizations and "scatter_plot" not in st.session_state.visualizations:
                        st.info("No relationship plots available.")
        
        elif st.session_state.active_tab == "eda":
            st.markdown('<p class="section-header">AI-Generated EDA Insights</p>', unsafe_allow_html=True)
            
            # Add generate button at the top
            if not st.session_state.eda_insights:
                if st.button("🔮 Generate EDA Insights", key="gen_eda"):
                    with st.spinner("Generating EDA insights using AI..."):
                        # Generate EDA insights using LLM
                        eda_insights = llm_inference.generate_eda_insights(st.session_state.dataset_info)
                        st.session_state.eda_insights = eda_insights
                        st.success("EDA insights generated!")
            
            if st.session_state.eda_insights:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(st.session_state.eda_insights)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a download link for the EDA insights
                st.download_button(
                    label="📥 Download EDA Insights",
                    data=st.session_state.eda_insights,
                    file_name="eda_insights.txt",
                    mime="text/plain"
                )
            else:
                st.info("Click the 'Generate EDA Insights' button to get AI-powered insights about your dataset.")
                
                # Show auto-generated basic insights
                if st.session_state.dataset_info:
                    # Display some basic automatically generated insights
                    st.markdown("### Automatically Generated Basic Insights")
                    
                    df = st.session_state.dataset_analyzer.df
                    
                    # Dataset size insight
                    rows, cols = df.shape
                    st.markdown(f'<div class="insight-box">📏 Dataset has {rows} rows and {cols} columns.</div>', unsafe_allow_html=True)
                    
                    # Data types insight
                    num_cols = len(st.session_state.dataset_info["numerical_columns"])
                    cat_cols = len(st.session_state.dataset_info["categorical_columns"])
                    st.markdown(f'<div class="insight-box">🔢 Contains {num_cols} numerical columns and {cat_cols} categorical columns.</div>', unsafe_allow_html=True)
                    
                    # Missing values insight
                    missing_cols = sum(1 for _, (count, _) in st.session_state.dataset_info["missing_values"].items() if count > 0)
                    if missing_cols > 0:
                        st.markdown(f'<div class="insight-box">⚠️ Found {missing_cols} columns with missing values.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="insight-box">✅ No missing values detected in the dataset.</div>', unsafe_allow_html=True)
                    
                    # Correlation insight
                    corr_text = st.session_state.dataset_info["correlations"]
                    if "Not enough" not in corr_text:
                        st.markdown('<div class="insight-box">🔗 Correlation analysis available. Check the "Relationships" tab in Dataset Overview.</div>', unsafe_allow_html=True)
        
        elif st.session_state.active_tab == "feature_engineering":
            st.markdown('<p class="section-header">AI-Generated Feature Engineering Recommendations</p>', unsafe_allow_html=True)
            
            # Add generate button at the top
            if not st.session_state.feature_engineering_recommendations:
                if st.button("🔮 Generate Feature Engineering Ideas", key="gen_fe"):
                    with st.spinner("Generating feature engineering recommendations using AI..."):
                        # Generate feature engineering recommendations using LLM
                        recommendations = llm_inference.generate_feature_engineering_recommendations(st.session_state.dataset_info)
                        st.session_state.feature_engineering_recommendations = recommendations
                        st.success("Feature engineering recommendations generated!")
            
            if st.session_state.feature_engineering_recommendations:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(st.session_state.feature_engineering_recommendations)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a download link for the recommendations
                st.download_button(
                    label="📥 Download Feature Engineering Recommendations",
                    data=st.session_state.feature_engineering_recommendations,
                    file_name="feature_engineering_recommendations.txt",
                    mime="text/plain"
                )
            else:
                st.info("Click the 'Generate Feature Engineering Ideas' button to get AI-powered feature engineering recommendations.")
                
                # Show auto-generated basic feature engineering ideas
                if st.session_state.dataset_loaded:
                    fe_ideas = st.session_state.dataset_analyzer.generate_feature_engineering_ideas()
                    
                    if fe_ideas:
                        st.markdown("### Basic Feature Engineering Ideas")
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        for idea in fe_ideas:
                            st.markdown(idea)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        elif st.session_state.active_tab == "data_quality":
            st.markdown('<p class="section-header">AI-Generated Data Quality Insights</p>', unsafe_allow_html=True)
            
            # Add generate button at the top
            if not st.session_state.data_quality_insights:
                if st.button("🔮 Generate Data Quality Insights", key="gen_dq"):
                    with st.spinner("Generating data quality insights using AI..."):
                        # Generate data quality insights using LLM
                        quality_insights = llm_inference.generate_data_quality_insights(st.session_state.dataset_info)
                        st.session_state.data_quality_insights = quality_insights
                        st.success("Data quality insights generated!")
            
            if st.session_state.data_quality_insights:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(st.session_state.data_quality_insights)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a download link for the data quality insights
                st.download_button(
                    label="📥 Download Data Quality Insights",
                    data=st.session_state.data_quality_insights,
                    file_name="data_quality_insights.txt",
                    mime="text/plain"
                )
            else:
                st.info("Click the 'Generate Data Quality Insights' button to get AI-powered data quality assessment.")
                
                # Show auto-generated preprocessing suggestions
                if st.session_state.dataset_loaded:
                    preprocessing_suggestions = st.session_state.dataset_analyzer.suggest_data_preprocessing()
                    
                    if preprocessing_suggestions:
                        st.markdown("### Data Preprocessing Suggestions")
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Missing values
                        if preprocessing_suggestions["missing_values"]:
                            st.markdown("#### Missing Values")
                            for suggestion in preprocessing_suggestions["missing_values"]:
                                st.markdown(f"- {suggestion}")
                        
                        # Outliers
                        if preprocessing_suggestions["outliers"]:
                            st.markdown("#### Potential Outliers")
                            for suggestion in preprocessing_suggestions["outliers"]:
                                st.markdown(f"- {suggestion}")
                        
                        # Numerical columns
                        if preprocessing_suggestions["numerical"]:
                            st.markdown("#### Numerical Features")
                            for suggestion in preprocessing_suggestions["numerical"]:
                                st.markdown(f"- {suggestion}")
                        
                        # Categorical columns
                        if preprocessing_suggestions["categorical"]:
                            st.markdown("#### Categorical Features")
                            for suggestion in preprocessing_suggestions["categorical"]:
                                st.markdown(f"- {suggestion}")
                        
                        # General suggestions
                        if preprocessing_suggestions["general"]:
                            st.markdown("#### General Recommendations")
                            for suggestion in preprocessing_suggestions["general"]:
                                st.markdown(f"- {suggestion}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        elif st.session_state.active_tab == "chat":
            st.markdown('<p class="section-header">Chat with Your Data</p>', unsafe_allow_html=True)
            
            st.markdown("""
            Ask questions about your dataset and get AI-powered answers based on the analysis.
            Examples:
            - What are the key patterns in this dataset?
            - Which columns have the most missing values?
            - What kind of feature engineering would be useful for this data?
            - How are the numerical variables distributed?
            """)
            
            # Chat interface
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Initialize chat history if not exists
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for entry in st.session_state.chat_history:
                if entry["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">{entry["content"]}</div>', unsafe_allow_html=True)
                else:  # AI
                    st.markdown(f'<div class="chat-message ai-message">{entry["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            user_question = st.text_input("Ask a question about your dataset:", key="chat_input")
            
            if user_question and st.button("Send", key="send_chat"):
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                with st.spinner("Generating answer..."):
                    answer = llm_inference.answer_dataset_question(user_question, st.session_state.dataset_info)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({"role": "ai", "content": answer})
                
                # Rerun to update the chat display
                st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display welcome message when no dataset is loaded
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## 👋 Welcome to the AI-Powered EDA & Feature Engineering Assistant
        
        This tool helps you analyze datasets and generate valuable insights with AI assistance.
        """)
        
        # Feature highlights with icons in a more appealing layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🔍 Key Features
            
            - **🤖 AI-Powered Analysis**: Get comprehensive EDA insights
            - **🧩 Feature Engineering**: Discover new features to improve your models
            - **⚠️ Data Quality Checks**: Identify and fix data issues
            """)
        with col2:
            st.markdown("""
            ### 📊 Visualizations
            
            - **📈 Automated Charts**: See your data through interactive visuals
            - **🔗 Correlation Analysis**: Discover relationships between variables
            - **💬 Chat Interface**: Ask questions about your data
            """)
        
        st.markdown("""
        ### 📝 How to Use
        
        1. Upload a CSV file using the sidebar
        2. The app will automatically analyze your data
        3. Navigate through the tabs to explore different insights
        4. Generate AI-powered recommendations with one click
        
        ### 🧠 Powered by
        
        - **Mistral-7B-Instruct-v0.3**: State-of-the-art LLM for data analysis
        - **Hugging Face**: For efficient API-based inference
        - **Streamlit**: For the interactive UI
        
        To get started, upload your CSV file using the sidebar on the left! 👈
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
