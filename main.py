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
import plotly.express as px
import numpy as np
# Import LangChain memory components
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# Import local modules
from eda_analysis import DatasetAnalyzer
from llm_inference import LLMInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page configuration - must be the first Streamlit command
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

# Add new functions to support the updated UI
def initialize_session_state():
    """Initialize session state variables needed for the application"""
    # Initialize session variables with appropriate defaults
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize conversation memory for LangChain
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    
    # For dataframe and related variables, ensure proper initialization
    # df should not be in session_state until a proper DataFrame is loaded
    if "descriptive_stats" not in st.session_state:
        st.session_state.descriptive_stats = None
        
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = []
        
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None
        
    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = None
        
    if "loading_insights" not in st.session_state:
        st.session_state.loading_insights = False
        
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = 'tab-overview'
        
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = ""
        
    # Logging initialization
    logger.info("Session state initialized")

def apply_custom_css():
    """Apply additional custom CSS that's not already in the main CSS block"""
    st.markdown("""
    <style>
        /* Base theme variables */
        :root {
            --primary: #4F46E5;
            --secondary: #06B6D4;
            --text-light: #F3F4F6;
            --text-muted: #9CA3AF;
            --bg-card: rgba(31, 41, 55, 0.7);
            --bg-dark: #111827;
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--bg-dark);
            color: var(--text-light);
        }
        
        /* Improve sidebar styling */
        .sidebar-header {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }
          /* Try to force change      
        div[class^="st-emotion-cache"] {
            background-color: #111827 !important;
        }*/ 
                
        div[data-testid="stBottomBlockContainer"] {
            background-color: #111827 !important;
        }        



        .sidebar-section {
            background: rgba(31, 41, 55, 0.4);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }
        
        .sidebar-footer {
            text-align: center;
            padding: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 3rem;
        }
        
        /* Feature Engineering Cards */
        .fe-cards-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.8rem;
            margin-top: 1rem;
        }
        
        .fe-card {
            background: rgba(31, 41, 55, 0.6);
            border-radius: 8px;
            padding: 0.8rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid rgba(99, 102, 241, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .fe-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 0;
        }
        
        .fe-card:hover::before {
            opacity: 0.1;
        }
        
        .fe-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-color: rgba(99, 102, 241, 0.3);
        }
        
        .fe-card-active {
            border-color: var(--primary);
            background: rgba(79, 70, 229, 0.1);
        }
        
        .fe-card-icon {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
            position: relative;
            z-index: 1;
        }
        
        .fe-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-light);
            position: relative;
            z-index: 1;
        }
        
        /* Tab content styling */
        .tab-title {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            position: relative;
            display: inline-block;
            color: var(--text-light);
        }
        
        .tab-title:after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 3px;
        }
        
        /* Navigation Tabs */
        .custom-tabs {
            display: flex;
            background: rgba(31, 41, 55, 0.6);
            border-radius: 12px;
            padding: 0.5rem;
            margin-bottom: 2rem;
            justify-content: space-between;
            overflow: hidden;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }
        
        .tab-item {
            flex: 1;
            text-align: center;
            padding: 0.8rem 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
            margin: 0 0.2rem;
        }
        
        .tab-item.active {
            background: rgba(79, 70, 229, 0.1);
        }
        
        .tab-item.active::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 3px;
        }
        
        .tab-item:hover {
            background: rgba(79, 70, 229, 0.05);
        }
        
        .tab-icon {
            font-size: 1.5rem;
            margin-bottom: 0.3rem;
        }
        
        .tab-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-light);
        }
        
        .tab-content-spacer {
            height: 1rem;
        }
        
        /* Card styling */
        .stats-card, .info-card, .chart-card {
            background: rgba(31, 41, 55, 0.3);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
            transition: all 0.3s ease;
        }
        
        .stats-card:hover, .info-card:hover, .chart-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            border-color: rgba(99, 102, 241, 0.3);
        }
        
        /* Dataset stats styling */
        .dataset-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            justify-content: center;
        }
        
        .stat-item {
            text-align: center;
            padding: 0.8rem;
            background: rgba(31, 41, 55, 0.6);
            border-radius: 8px;
            min-width: 80px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
        }
        
        /* Chart styling */
        .chart-container {
            margin-top: 1.5rem;
        }
        
        .chart-card h3 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: var(--text-light);
        }
        
        .stat-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .stat-pair {
            display: flex;
            justify-content: space-between;
            padding: 0.3rem 0.5rem;
            background: rgba(31, 41, 55, 0.4);
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .stat-pair span {
            color: var(--text-muted);
        }
        
        .stat-pair strong {
            color: var(--text-light);
        }
        
        /* Filter container */
        .filter-container {
            background: rgba(31, 41, 55, 0.3);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }
        
        /* AI Insights styling */
        .insights-container {
            margin-top: 1rem;
        }
        
        .insights-category {
            margin-top: 0.5rem;
        }
        
        .insight-card {
            background: rgba(31, 41, 55, 0.3);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
            display: flex;
            align-items: flex-start;
        }
        
        .insight-content {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }
        
        .insight-icon {
            font-size: 1.5rem;
            margin-top: 0.1rem;
        }
        
        .insight-text {
            flex: 1;
            line-height: 1.5;
        }
        
        .generate-insights-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 3rem 0;
        }
        
        .placeholder-card {
            background: rgba(31, 41, 55, 0.3);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(99, 102, 241, 0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        
        .placeholder-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: float 3s ease-in-out infinite;
        }
        
        .placeholder-text {
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        
        .loading-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        
        .loading-pulse {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            animation: pulse-animation 1.5s ease infinite;
        }
        
        @keyframes pulse-animation {
            0% {
                transform: scale(0.6);
                opacity: 0.5;
            }
            50% {
                transform: scale(1);
                opacity: 1;
            }
            100% {
                transform: scale(0.6);
                opacity: 0.5;
            }
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Button styling */
        button[kind="primary"] {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15) !important;
        }
        
        button[kind="secondary"] {
            background: rgba(79, 70, 229, 0.1) !important;
            color: var(--text-light) !important;
            border: 1px solid rgba(79, 70, 229, 0.3) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        button[kind="secondary"]:hover {
            background: rgba(79, 70, 229, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Override Streamlit default button styles */
        .stButton>button {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Chat interface styling */
        .chat-interface-container {
            padding: 1rem 0;
            margin-bottom: 100px;
            position: relative;
        }
        
        .chat-messages {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .chat-message-user, .chat-message-ai {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .chat-message-user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border-bottom-right-radius: 0;
            margin-left: auto;
        }
        
        .chat-message-ai {
            align-self: flex-start;
            background: var(--bg-card);
            color: var(--text-light);
            border-bottom-left-radius: 0;
            margin-right: auto;
        }
        
        .chat-input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 1.5rem;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 1.5rem 0;
        }
        
        .chat-suggestion {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 30px;
            padding: 8px 15px;
            font-size: 0.9rem;
            color: var(--text-light);
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            margin-bottom: 8px;
        }
        
        .chat-suggestion:hover {
            background: rgba(99, 102, 241, 0.2);
            transform: translateY(-2px);
        }
        
        /* Expander styling */
        .st-expander {
            background: rgba(31, 41, 55, 0.2) !important;
            border-radius: 8px !important;
            margin-bottom: 1rem !important;
            border: 1px solid rgba(99, 102, 241, 0.1) !important;
        }
        
        /* Streamlit widget styling */
        div[data-testid="stForm"] {
            background: rgba(31, 41, 55, 0.2) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            border: 1px solid rgba(99, 102, 241, 0.1) !important;
        }
        
        .stSelectbox>div>div {
            background: rgba(31, 41, 55, 0.4) !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 8px !important;
        }
        
        .stTextInput>div>div>input {
            background: rgba(31, 41, 55, 0.4) !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 8px !important;
            color: var(--text-light) !important;
            padding: 1rem !important;
        }
        
        /* Streamlit multiselect dropdown styling */
        div[data-baseweb="popover"] {
            background: var(--bg-dark) !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 8px !important;
        }
        
        div[data-baseweb="menu"] {
            background: var(--bg-dark) !important;
        }
        
        div[role="listbox"] {
            background: var(--bg-dark) !important;
        }
        
        /* Fix for the upload button */
        .stFileUploader > div {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .stFileUploader > div > button {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border: none !important;
            width: 100%;
            margin-top: 1rem;
        }
        
        /* Fix for tab content spacing */
        .tab-content {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(31, 41, 55, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def generate_ai_insights():
    """Generate AI-powered insights about the dataset"""
    # Make sure we have a dataframe to analyze
    if 'df' not in st.session_state:
        logger.warning("Cannot generate AI insights: No dataframe in session state")
        return {}
    
    df = st.session_state.df
    insights = {}
    
    # Try to use the LLM for insights generation first
    try:
        if llm_inference is not None:
            # Create dataset_info dictionary for LLM
            num_rows, num_cols = df.shape
            num_numerical = len(df.select_dtypes(include=['number']).columns)
            num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
            num_missing = df.isnull().sum().sum()
            
            # Format missing values for better readability
            missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
            missing_values = {}
            for col in missing_cols.index:
                count = missing_cols[col]
                percent = round(count / len(df) * 100, 2)
                missing_values[col] = (count, percent)
            
            # Get numerical columns and their correlations if applicable
            num_cols = df.select_dtypes(include=['number']).columns
            correlations = "No numerical columns to calculate correlations."
            if len(num_cols) > 1:
                # Calculate correlations
                corr_matrix = df[num_cols].corr()
                # Get top correlations (absolute values)
                corr_pairs = []
                for i in range(len(num_cols)):
                    for j in range(i):
                        val = corr_matrix.iloc[i, j]
                        if abs(val) > 0.5:  # Only show strong correlations
                            corr_pairs.append((num_cols[i], num_cols[j], val))
                
                # Sort by absolute correlation and format
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    formatted_corrs = []
                    for col1, col2, val in corr_pairs[:5]:  # Top 5
                        formatted_corrs.append(f"{col1} and {col2}: {val:.3f}")
                    correlations = "\n".join(formatted_corrs)
            
            dataset_info = {
                "shape": f"{num_rows} rows, {num_cols} columns",
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": missing_values,
                "basic_stats": df.describe().to_string(),
                "correlations": correlations,
                "sample_data": df.head(5).to_string()
            }
            
            # Generate EDA insights with better error handling
            logger.info("Requesting EDA insights from LLM")
            try:
                eda_insights = llm_inference.generate_eda_insights(dataset_info)
                
                if eda_insights and isinstance(eda_insights, str) and len(eda_insights) > 50:
                    # Clean and format the response
                    eda_insights = eda_insights.strip()
                    insights["EDA Insights"] = [eda_insights]
                    logger.info("Successfully generated EDA insights")
                else:
                    logger.warning(f"EDA insights response was invalid: {type(eda_insights)}, length: {len(eda_insights) if isinstance(eda_insights, str) else 'N/A'}")
            except Exception as e:
                logger.error(f"Error generating EDA insights: {str(e)}")
            
            # Generate feature engineering recommendations
            if "EDA Insights" in insights:  # Only proceed if EDA worked
                logger.info("Requesting feature engineering recommendations from LLM")
                try:
                    fe_insights = llm_inference.generate_feature_engineering_recommendations(dataset_info)
                    
                    if fe_insights and isinstance(fe_insights, str) and len(fe_insights) > 50:
                        fe_insights = fe_insights.strip()
                        insights["Feature Engineering Recommendations"] = [fe_insights]
                        logger.info("Successfully generated feature engineering recommendations")
                    else:
                        logger.warning(f"Feature engineering response was invalid: {type(fe_insights)}, length: {len(fe_insights) if isinstance(fe_insights, str) else 'N/A'}")
                except Exception as e:
                    logger.error(f"Error generating feature engineering recommendations: {str(e)}")
            
                # Generate data quality insights
                logger.info("Requesting data quality insights from LLM")
                try:
                    dq_insights = llm_inference.generate_data_quality_insights(dataset_info)
                    
                    if dq_insights and isinstance(dq_insights, str) and len(dq_insights) > 50:
                        dq_insights = dq_insights.strip()
                        insights["Data Quality Insights"] = [dq_insights]
                        logger.info("Successfully generated data quality insights")
                    else:
                        logger.warning(f"Data quality response was invalid: {type(dq_insights)}, length: {len(dq_insights) if isinstance(dq_insights, str) else 'N/A'}")
                except Exception as e:
                    logger.error(f"Error generating data quality insights: {str(e)}")
            
            # If we have at least one type of insights, consider it a success
            if insights:
                # Mark that the insights are loaded
                st.session_state['loading_insights'] = False
                logger.info("Successfully generated AI insights using LLM")
                return insights
            
            logger.warning("All LLM generated insights failed or were too short. Falling back to template insights.")
        else:
            logger.warning("LLM inference is not available. Falling back to template insights.")
    except Exception as e:
        logger.error(f"Error in generate_ai_insights(): {str(e)}. Falling back to template insights.")
    
    # If LLM fails or is not available, generate template-based insights
    logger.info("Falling back to template-based insights generation")
    
    # Add missing values insights
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_cols = missing_data[missing_data > 0]
    
    missing_insights = []
    if len(missing_cols) > 0:
        missing_insights.append(f"Found {len(missing_cols)} columns with missing values.")
        for col in missing_cols.index[:3]:  # Show details for top 3
            missing_insights.append(f"Column '{col}' has {missing_data[col]} missing values ({missing_percent[col]:.2f}%).")
        
        if len(missing_cols) > 3:
            missing_insights.append(f"And {len(missing_cols) - 3} more columns have missing values.")
            
        # Add recommendation
        if any(missing_percent > 50):
            high_missing = missing_percent[missing_percent > 50].index.tolist()
            missing_insights.append(f"Consider dropping columns with >50% missing values: {', '.join(high_missing[:3])}.")
        else:
            missing_insights.append("Consider using imputation techniques for columns with missing values.")
    else:
        missing_insights.append("No missing values found in the dataset. Great job!")
    
    insights["Missing Values Analysis"] = missing_insights
    
    # Add distribution insights
    num_cols = df.select_dtypes(include=['number']).columns
    dist_insights = []
    
    if len(num_cols) > 0:
        for col in num_cols[:3]:  # Analyze top 3 numeric columns
            # Check for skewness
            skew = df[col].skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                dist_insights.append(f"Column '{col}' is {direction}-skewed (skewness: {skew:.2f}). Consider log transformation.")
            
            # Check for outliers using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
            
            if outliers > 0:
                pct = (outliers / len(df)) * 100
                dist_insights.append(f"Column '{col}' has {outliers} outliers ({pct:.2f}%). Consider outlier treatment.")
        
        if len(num_cols) > 3:
            dist_insights.append(f"Additional {len(num_cols) - 3} numerical columns not analyzed here.")
    else:
        dist_insights.append("No numerical columns found for distribution analysis.")
    
    insights["Distribution Insights"] = dist_insights
    
    # Add correlation insights
    corr_insights = []
    if len(num_cols) > 1:
        # Calculate correlation
        corr_matrix = df[num_cols].corr()
        high_corr = []
        
        # Find high correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            corr_insights.append(f"Found {len(high_corr)} pairs of highly correlated features.")
            for col1, col2, corr_val in high_corr[:3]:  # Show top 3
                corr_direction = "positively" if corr_val > 0 else "negatively"
                corr_insights.append(f"'{col1}' and '{col2}' are strongly {corr_direction} correlated (r={corr_val:.2f}).")
            
            if len(high_corr) > 3:
                corr_insights.append(f"And {len(high_corr) - 3} more highly correlated pairs found.")
            
            corr_insights.append("Consider removing some highly correlated features to reduce dimensionality.")
        else:
            corr_insights.append("No strong correlations found between features.")
    else:
        corr_insights.append("Need at least 2 numerical columns to analyze correlations.")
    
    insights["Correlation Analysis"] = corr_insights
    
    # Add feature engineering recommendations
    fe_insights = []
    
    # Check for date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass
    
    if date_cols:
        fe_insights.append(f"Found {len(date_cols)} potential date columns: {', '.join(date_cols[:3])}.")
        fe_insights.append("Consider extracting year, month, day, weekday from these columns.")
    
    # Check for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        fe_insights.append(f"Found {len(cat_cols)} categorical columns.")
        fe_insights.append("Consider one-hot encoding or label encoding for categorical features.")
        
        # Check for high cardinality
        high_card_cols = []
        for col in cat_cols:
            if df[col].nunique() > 10:
                high_card_cols.append((col, df[col].nunique()))
        
        if high_card_cols:
            fe_insights.append(f"Some categorical columns have high cardinality:")
            for col, card in high_card_cols[:2]:
                fe_insights.append(f"Column '{col}' has {card} unique values. Consider grouping less common categories.")
    
    # Suggest polynomial features if few numeric features
    if 1 < len(num_cols) < 5:
        fe_insights.append("Consider creating polynomial features or interaction terms between numerical features.")
    
    insights["Feature Engineering Recommendations"] = fe_insights
    
    # Add a slight delay to simulate processing
    time.sleep(1)
    
    # Mark that the insights are loaded
    st.session_state['loading_insights'] = False
    logger.info("Template-based insights generation completed")
    
    return insights

def display_chat_interface():
    """Display a chat interface for interacting with the data"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="tab-title">💬 Chat with Your Data</h2>', unsafe_allow_html=True)
    
    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Make sure we have data to chat about
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("No dataset loaded. Please upload a CSV file to chat with your data.")
        
        # Show a preview of chat capabilities
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3>What can I help you with?</h3>
            <p>Once you upload a dataset, you can ask questions like:</p>
            <ul>
                <li>What patterns do you see in my data?</li>
                <li>How many missing values are there?</li>
                <li>What feature engineering would you recommend?</li>
                <li>Show me the distribution of a specific column</li>
                <li>What are the correlations between features?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Add a button to clear chat history
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            # Reset conversation memory
            if "conversation_memory" in st.session_state:
                st.session_state.conversation_memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True
                )
            logger.info("Chat history and memory cleared")
            st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # If no chat history, show some example questions
    if not st.session_state.chat_history:
        st.info("Ask me anything about your dataset! I can help you understand patterns, identify issues, and suggest improvements.")
        
        st.markdown("### Example questions you can ask:")
        
        # Create a grid of example questions using columns
        col1, col2 = st.columns(2)
        
        with col1:
            example_questions = [
                "What are the key patterns in this dataset?",
                "Which columns have missing values?",
                "What kind of feature engineering would help?"
            ]
            
            for i, question in enumerate(example_questions):
                if st.button(question, key=f"example_q_{i}"):
                    process_chat_message(question)
                    st.rerun()
                    
        with col2:
            more_questions = [
                "How are the numerical variables distributed?",
                "What are the strongest correlations?",
                "How can I prepare this data for modeling?"
            ]
            
            for i, question in enumerate(more_questions):
                if st.button(question, key=f"example_q_{i+3}"):
                    process_chat_message(question)
                    st.rerun()
    
    # Input area for new messages
    user_input = st.chat_input("Ask a question about your data...", key="chat_input")
    
    if user_input:
        # Add user message to chat history
        process_chat_message(user_input)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_descriptive_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="tab-title">📊 Descriptive Statistics</h2>', unsafe_allow_html=True)
    
    # Make sure we access the data from session state
    if 'df' not in st.session_state or 'descriptive_stats' not in st.session_state:
        st.error("No dataset loaded. Please upload a CSV file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    descriptive_stats = st.session_state.descriptive_stats
    
    # Display descriptive statistics in a more visually appealing way
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Style the dataframe
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.subheader("Numerical Summary")
        st.dataframe(descriptive_stats.style.background_gradient(cmap='Blues', axis=0)
                    .format(precision=2, na_rep="Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        
        # Display dataset information in a cleaner format
        total_rows = df.shape[0]
        total_cols = df.shape[1]
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        date_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        st.markdown(f"""
        <div class="dataset-stats">
            <div class="stat-item">
                <div class="stat-value">{total_rows:,}</div>
                <div class="stat-label">Rows</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_cols}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{numeric_cols}</div>
                <div class="stat-label">Numerical</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{cat_cols}</div>
                <div class="stat-label">Categorical</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{date_cols}</div>
                <div class="stat-label">Date/Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add missing values information with visualization
    st.markdown('<div class="stats-card">', unsafe_allow_html=True)
    st.subheader("Missing Values")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage (%)': missing_percent.round(2)
        })
        missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        
        if not missing_data.empty:
            st.dataframe(missing_data.style.background_gradient(cmap='Reds', subset=['Percentage (%)'])
                        .format({'Percentage (%)': '{:.2f}%'}), use_container_width=True)
        else:
            st.success("No missing values found in the dataset! 🎉")
    
    with col2:
        if not missing_data.empty:
            # Create a horizontal bar chart for missing values
            fig = px.bar(missing_data, 
                        x='Percentage (%)', 
                        y=missing_data.index, 
                        orientation='h',
                        color='Percentage (%)',
                        color_continuous_scale='Reds',
                        title='Missing Values by Column')
            
            fig.update_layout(
                height=max(350, len(missing_data) * 30),
                xaxis_title='Missing (%)',
                yaxis_title='',
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_distribution_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="tab-title">📈 Data Distribution</h2>', unsafe_allow_html=True)
    
    # Make sure we access the data from session state
    if 'df' not in st.session_state:
        st.error("No dataset loaded. Please upload a CSV file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Add filters for better UX
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Violin Plot", "Distribution Plot"],
            key="chart_type_select"
        )
    
    with col2:
        if chart_type != "Distribution Plot":
            column_type = "Numerical" if chart_type in ["Histogram", "Box Plot", "Violin Plot"] else "Categorical"
            columns_to_show = df.select_dtypes(include=['number']).columns.tolist() if column_type == "Numerical" else df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            selected_columns = st.multiselect(
                f"Select {column_type} Columns to Visualize",
                options=columns_to_show,
                default=columns_to_show[:min(3, len(columns_to_show))],
                key="column_select"
            )
        else:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            selected_columns = st.multiselect(
                "Select Numerical Columns",
                options=num_cols,
                default=num_cols[:min(3, len(num_cols))],
                key="column_select"
            )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display selected charts
    if selected_columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if chart_type == "Histogram":
            col1, col2 = st.columns([3, 1])
            with col2:
                bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, key="hist_bins")
                kde = st.checkbox("Show KDE", value=True, key="show_kde")
            
            with col1:
                pass
            
            # Display histograms with better styling
            for column in selected_columns:
                st.markdown(f'<div class="chart-card"><h3>{column}</h3>', unsafe_allow_html=True)
                fig = px.histogram(df, x=column, nbins=bins, 
                                 title=f"Histogram of {column}",
                                 marginal="box" if kde else None,
                                 color_discrete_sequence=['rgba(99, 102, 241, 0.7)'])
                
                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title=column,
                    yaxis_title="Frequency",
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show basic statistics 
                stats = df[column].describe().to_dict()
                st.markdown(f"""
                <div class="stat-summary">
                    <div class="stat-pair"><span>Mean:</span> <strong>{stats['mean']:.2f}</strong></div>
                    <div class="stat-pair"><span>Median:</span> <strong>{stats['50%']:.2f}</strong></div>
                    <div class="stat-pair"><span>Std Dev:</span> <strong>{stats['std']:.2f}</strong></div>
                    <div class="stat-pair"><span>Min:</span> <strong>{stats['min']:.2f}</strong></div>
                    <div class="stat-pair"><span>Max:</span> <strong>{stats['max']:.2f}</strong></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
        elif chart_type == "Box Plot":
            for column in selected_columns:
                st.markdown(f'<div class="chart-card"><h3>{column}</h3>', unsafe_allow_html=True)
                fig = px.box(df, y=column, title=f"Box Plot of {column}",
                           color_discrete_sequence=['rgba(99, 102, 241, 0.7)'])
                
                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title=column
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier information
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                
                st.markdown(f"""
                <div class="stat-summary">
                    <div class="stat-pair"><span>Q1 (25%):</span> <strong>{q1:.2f}</strong></div>
                    <div class="stat-pair"><span>Median:</span> <strong>{df[column].median():.2f}</strong></div>
                    <div class="stat-pair"><span>Q3 (75%):</span> <strong>{q3:.2f}</strong></div>
                    <div class="stat-pair"><span>IQR:</span> <strong>{iqr:.2f}</strong></div>
                    <div class="stat-pair"><span>Outliers:</span> <strong>{len(outliers)}</strong> ({(len(outliers)/len(df)*100):.2f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
        elif chart_type == "Violin Plot":
            for column in selected_columns:
                st.markdown(f'<div class="chart-card"><h3>{column}</h3>', unsafe_allow_html=True)
                fig = px.violin(df, y=column, box=True, points="all", title=f"Violin Plot of {column}",
                              color_discrete_sequence=['rgba(99, 102, 241, 0.7)'])
                
                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title=column
                )
                
                fig.update_traces(marker=dict(size=3, opacity=0.5))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
        elif chart_type == "Distribution Plot":
            if len(selected_columns) >= 2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                chart_options = st.radio(
                    "Select Distribution Plot Type",
                    ["Scatter Plot", "Correlation Heatmap"],
                    horizontal=True
                )
                
                if chart_options == "Scatter Plot":
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        x_axis = st.selectbox("X-axis", options=selected_columns, index=0)
                        y_axis = st.selectbox("Y-axis", options=selected_columns, index=min(1, len(selected_columns)-1))
                        color_option = st.selectbox("Color by", options=["None"] + df.columns.tolist())
                    
                    with col1:
                        if color_option != "None":
                            fig = px.scatter(df, x=x_axis, y=y_axis, 
                                           color=color_option,
                                           title=f"{y_axis} vs {x_axis} (colored by {color_option})",
                                           opacity=0.7,
                                           marginal_x="histogram", marginal_y="histogram")
                        else:
                            fig = px.scatter(df, x=x_axis, y=y_axis, 
                                           title=f"{y_axis} vs {x_axis}",
                                           opacity=0.7,
                                           marginal_x="histogram", marginal_y="histogram")
                        
                        fig.update_layout(
                            template="plotly_white",
                            height=600,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_options == "Correlation Heatmap":
                    # Calculate correlation matrix
                    corr_matrix = df[selected_columns].corr()
                    
                    # Create heatmap
                    fig = px.imshow(corr_matrix, 
                                   text_auto=".2f",
                                   color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1,
                                   title="Correlation Heatmap")
                    
                    fig.update_layout(
                        template="plotly_white",
                        height=600,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show highest correlations
                    corr_df = corr_matrix.stack().reset_index()
                    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
                    corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
                    corr_df = corr_df.sort_values('Correlation', ascending=False).head(5)
                    
                    st.markdown("##### Top 5 Highest Correlations")
                    st.dataframe(corr_df.style.background_gradient(cmap='Blues')
                                .format({'Correlation': '{:.2f}'}), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please select at least 2 numerical columns to see distribution plots")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please select at least one column to visualize")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_ai_insights_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="tab-title">🧠 AI-Generated Insights</h2>', unsafe_allow_html=True)
    
    # Make sure we access the data from session state
    if 'df' not in st.session_state:
        st.error("No dataset loaded. Please upload a CSV file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if st.session_state.get('loading_insights', False):
        with st.spinner("Generating AI insights about your data..."):
            st.markdown('<div class="loading-container"><div class="loading-pulse"></div></div>', unsafe_allow_html=True)
            time.sleep(0.1)  # Small delay to ensure UI updates
    
    # AI insights section
    if 'ai_insights' in st.session_state and st.session_state.ai_insights and len(st.session_state.ai_insights) > 0:
        insights = st.session_state.ai_insights
        
        st.markdown('<div class="insights-container">', unsafe_allow_html=True)
        
        for i, (category, insight_list) in enumerate(insights.items()):
            with st.expander(f"{category}", expanded=i < 2):
                st.markdown('<div class="insights-category">', unsafe_allow_html=True)
                
                # Check if the insights are from LLM (single string) or template (list of strings)
                if len(insight_list) == 1 and isinstance(insight_list[0], str) and len(insight_list[0]) > 100:
                    # This is likely an LLM-generated insight (single long string)
                    st.markdown(insight_list[0])
                else:
                    # Template-based insights (list of strings)
                    for insight in insight_list:
                        st.markdown(f"""
                        <div class="insight-card">
                            <div class="insight-content">
                                <div class="insight-icon">💡</div>
                                <div class="insight-text">{insight}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add regenerate button
        st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("Regenerate Insights", key="regenerate_insights"):
            st.session_state['loading_insights'] = True
            st.session_state['ai_insights'] = None
            logger.info("User requested regeneration of AI insights")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if not st.session_state.get('loading_insights', False):
            # Show generate button if insights are not loading and not available
            st.markdown('<div class="generate-insights-container">', unsafe_allow_html=True)
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon">🧠</div>
                <div class="placeholder-text">Generate AI-powered insights about your dataset to discover patterns, anomalies, and suggestions for feature engineering.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Insights", key="generate_insights"):
                st.session_state['loading_insights'] = True
                logger.info("User initiated AI insights generation")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_welcome_page():
    """Display a welcome page with information about the application"""
    # Use Streamlit columns and components instead of raw HTML
    st.title("Welcome to AI-Powered EDA & Feature Engineering Assistant")
    
    st.write("""
    Upload your CSV dataset and leverage the power of AI to analyze, visualize, and improve your data.
    This tool helps you understand your data better and prepare it for machine learning models.
    """)
    
    # Feature cards
    st.subheader("Key Features")
    
    # Use Streamlit columns to create a grid layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Exploratory Data Analysis")
        st.write("Quickly understand your dataset with automatic statistical analysis and visualizations")
        
        st.markdown("#### 🧠 AI-Powered Insights")
        st.write("Get intelligent recommendations about patterns, anomalies, and opportunities in your data")
        
        st.markdown("#### ⚡ Feature Engineering")
        st.write("Transform and enhance your features to improve machine learning model performance")
    
    with col2:
        st.markdown("#### 📈 Interactive Visualizations")
        st.write("Explore distributions, relationships, and outliers with dynamic charts")
        
        st.markdown("#### 💬 Chat Interface")
        st.write("Ask questions about your data and get AI-powered answers in natural language")
        
        st.markdown("#### 🔄 Data Transformation")
        st.write("Clean, transform, and prepare your data for modeling with guided workflows")
    
    # Usage section
    st.subheader("How to use")
    
    st.markdown("""
    1. **Upload** your CSV dataset using the sidebar on the left
    2. **Explore** automatically generated statistics and visualizations
    3. **Generate** AI insights to better understand your data
    4. **Chat** with AI to ask specific questions about your dataset
    5. **Transform** your features based on recommendations
    """)
    
    # Powered by section
    st.subheader("Powered by")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**llama3-8b-8192**")
    with cols[1]:
        st.markdown("**Groq API**")
    with cols[2]:
        st.markdown("**Streamlit**")
    
    # Upload prompt
    st.info("👈 Please upload a CSV file using the sidebar to get started")

def display_relationships_tab():
    """Display correlations and relationships between variables"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="tab-title">🔄 Relationships & Correlations</h2>', unsafe_allow_html=True)
    
    # Make sure we have data to visualize
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("No dataset loaded. Please upload a CSV file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Select numerical columns for correlation analysis
    num_cols = df.select_dtypes(include=['number']).columns
    
    if len(num_cols) < 2:
        st.warning("At least 2 numerical columns are needed for correlation analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Correlation matrix heatmap
    st.subheader("Correlation Matrix")
    
    # Calculate correlation
    corr_matrix = df[num_cols].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=600,
        width=800,
        title_font_size=20,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top correlations
    st.subheader("Top Correlations")
    
    # Extract and format correlations
    corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i):
            corr_pairs.append({
                'Feature 1': num_cols[i],
                'Feature 2': num_cols[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    # Convert to dataframe and sort
    corr_df = pd.DataFrame(corr_pairs)
    sorted_corr = corr_df.sort_values('Correlation', key=abs, ascending=False).head(10)
    
    # Show table with styled background
    st.dataframe(
        sorted_corr.style.background_gradient(cmap='RdBu_r', subset=['Correlation'])
            .format({'Correlation': '{:.3f}'}),
        use_container_width=True
    )
    
    # Scatter plot matrix
    st.subheader("Scatter Plot Matrix")
    
    # Let user choose columns
    selected_cols = st.multiselect(
        "Select columns for scatter plot matrix (max 5 recommended)",
        options=num_cols,
        default=num_cols[:min(4, len(num_cols))]
    )
    
    if selected_cols:
        if len(selected_cols) > 5:
            st.warning("More than 5 columns may make the plot hard to read.")
        
        color_col = st.selectbox("Color by", options=["None"] + df.columns.tolist())
        
        # Only pass the color parameter if not "None"
        if color_col != "None":
            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                color=color_col,
                opacity=0.7,
                title="Scatter Plot Matrix"
            )
        else:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                opacity=0.7,
                title="Scatter Plot Matrix"
            )
        
        fig.update_layout(
            height=700,
            title_font_size=18,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_chat_message(user_message):
    """Process a user message in the chat interface"""
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    
    # Generate a response from the AI
    if 'df' in st.session_state and st.session_state.df is not None:
        # Try to use LLM if available, otherwise fall back to templates
        try:
            if llm_inference is not None:
                # Create a prompt about the dataset
                df = st.session_state.df
                
                # Get basic dataset info
                num_rows, num_cols = df.shape
                num_numerical = len(df.select_dtypes(include=['number']).columns)
                num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
                num_missing = df.isnull().sum().sum()
                missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
                
                # Format missing values for better readability
                missing_values = {}
                for col in missing_cols.index:
                    count = missing_cols[col]
                    percent = round(count / len(df) * 100, 2)
                    missing_values[col] = (count, percent)
                
                # Get correlations for numerical columns
                num_cols = df.select_dtypes(include=['number']).columns
                correlations = "No numerical columns to calculate correlations."
                if len(num_cols) > 1:
                    # Calculate correlations
                    corr_matrix = df[num_cols].corr()
                    # Get top 5 correlations (absolute values)
                    corr_pairs = []
                    for i in range(len(num_cols)):
                        for j in range(i):
                            val = corr_matrix.iloc[i, j]
                            if abs(val) > 0.5:  # Only show strong correlations
                                corr_pairs.append((num_cols[i], num_cols[j], val))
                    
                    # Sort by absolute correlation and format
                    if corr_pairs:
                        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                        formatted_corrs = []
                        for col1, col2, val in corr_pairs[:5]:  # Top 5
                            formatted_corrs.append(f"{col1} and {col2}: {val:.3f}")
                        correlations = "\n".join(formatted_corrs)
                
                # Create dataset_info dictionary for LLM
                dataset_info = {
                    "shape": f"{num_rows} rows, {num_cols} columns",
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": missing_values,
                    "basic_stats": df.describe().to_string(),
                    "correlations": correlations,
                    "sample_data": df.head(5).to_string()
                }
                
                # Generate response using LLM with memory
                logger.info(f"Sending question to LLM with memory: {user_message}")
                
                # Convert chat history to LangChain format for the memory object if needed
                if len(st.session_state.chat_history) > 1 and "conversation_memory" in st.session_state:
                    # Use the memory-enabled version to maintain conversation context
                    response = llm_inference.answer_with_memory(
                        user_message, 
                        dataset_info,
                        st.session_state.conversation_memory
                    )
                else:
                    # If it's the first message, just use the regular question answering
                    response = llm_inference.answer_dataset_question(user_message, dataset_info)
                    
                    # Initialize the memory with this first exchange
                    if "conversation_memory" in st.session_state:
                        st.session_state.conversation_memory.save_context(
                            {"input": user_message},
                            {"output": response}
                        )
                
                # Log the raw response for debugging
                logger.info(f"Raw LLM response: {response[:100]}...")
                
                # If response is not empty and is a valid string
                if response and isinstance(response, str) and len(response) > 10:
                    # Clean up the response if needed
                    cleaned_response = response.strip()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
                    return
                else:
                    logger.warning(f"LLM response too short or invalid: {response}")
                    raise Exception("LLM response too short or invalid")
            else:
                raise Exception("LLM not available")
                
        except Exception as e:
            logger.warning(f"Error using LLM for chat response: {str(e)}. Falling back to templates.")
            # Fall back happens below
    
    # If we're here, either there's no dataframe, LLM failed, or response was invalid
    # Use template-based responses as fallback
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Simple response templates
        responses = {
            "missing": f"I found {df.isnull().sum().sum()} missing values across the dataset. The columns with the most missing values are: {df.isnull().sum().sort_values(ascending=False).head(3).index.tolist()}.",
            "pattern": "Looking at the data, I can see several interesting patterns. The numerical features show varied distributions, and there might be some correlations worth exploring further.",
            "feature": "Based on the data, I'd recommend feature engineering steps like handling missing values, encoding categorical variables, and possibly creating interaction terms for highly correlated features.",
            "distribution": f"The numerical variables show different distributions. Some appear to be normally distributed while others show skewness. Let me know if you want to see visualizations for specific columns.",
            "correlation": "I detected several strong correlations in the dataset. You might want to look at the correlation heatmap in the Relationships tab for more details.",
            "prepare": "To prepare this data for modeling, I suggest: 1) Handling missing values, 2) Encoding categorical variables, 3) Feature scaling, and 4) Possibly dimensionality reduction if you have many features."
        }
        
        # Simple keyword matching for demo purposes
        if "missing" in user_message.lower():
            response = responses["missing"]
        elif "pattern" in user_message.lower():
            response = responses["pattern"]
        elif "feature" in user_message.lower() or "engineering" in user_message.lower():
            response = responses["feature"]
        elif "distribut" in user_message.lower():
            response = responses["distribution"]
        elif "correlat" in user_message.lower() or "relation" in user_message.lower():
            response = responses["correlation"]
        elif "prepare" in user_message.lower() or "model" in user_message.lower():
            response = responses["prepare"]
        else:
            # Generic response
            response = "I analyzed your dataset and found some interesting insights. You can explore different aspects of your data using the tabs above. Is there anything specific you'd like to know about your data?"
    else:
        response = "Please upload a dataset first so I can analyze it and answer your questions."
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    """Main function to run the application"""
    # Initialize session state at the beginning
    initialize_session_state()
    
    # Apply CSS styling
    apply_custom_css()
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.markdown('<div class="sidebar-header">AI-Powered EDA & Feature Engineering</div>', unsafe_allow_html=True)
        
        # File uploader
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('### Upload Dataset')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load example dataset
        with st.expander("Or use an example dataset"):
            example_datasets = {
                "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                "Titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
                "Diamonds": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
            }
            selected_example = st.selectbox("Select example dataset", list(example_datasets.keys()))
            if st.button("Load Example", key="load_example_btn"):
                try:
                    # Load the selected example dataset
                    df = pd.read_csv(example_datasets[selected_example])
                    
                    # Verify we have a valid dataframe
                    if df is not None and not df.empty:
                        st.session_state['df'] = df
                        st.session_state['descriptive_stats'] = df.describe()
                        st.session_state['dataset_name'] = selected_example
                        st.success(f"Loaded {selected_example} dataset!")
                    else:
                        st.error(f"The {selected_example} dataset appears to be empty.")
                except Exception as e:
                    st.error(f"Error loading example dataset: {str(e)}")
        
        # Only show these sections if a dataset is loaded
        if 'df' in st.session_state:
            # Dataset Info
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown(f'### Dataset Info: {st.session_state.get("dataset_name", "Uploaded Data")}')
            df = st.session_state.df
            # Add check to ensure df is not None before accessing shape
            if df is not None:
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            else:
                st.error("Dataset is loaded but appears to be empty.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Column filters
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('### Column Filters')
            if df is not None:
                selected_columns = st.multiselect("Select columns to analyze", 
                                                options=df.columns.tolist(),
                                                default=df.columns.tolist())
                
                if len(selected_columns) > 0:
                    st.session_state['selected_columns'] = selected_columns
                    st.session_state['filtered_df'] = df[selected_columns]
                else:
                    st.session_state['selected_columns'] = df.columns.tolist()
                    st.session_state['filtered_df'] = df
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature Engineering options with Streamlit buttons instead of JavaScript
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('### Feature Engineering')
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Missing Values", key="missing_values_btn"):
                    st.session_state['fe_selected'] = 'missing_values'
            
            with col2:
                if st.button("Encode Categorical", key="encode_cat_btn"):
                    st.session_state['fe_selected'] = 'encode_categorical'
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Scale Features", key="scale_features_btn"):
                    st.session_state['fe_selected'] = 'scale_features'
            
            with col2:
                if st.button("Transform", key="transform_btn"):
                    st.session_state['fe_selected'] = 'transform'
            
            # Display currently selected feature engineering option
            if 'fe_selected' in st.session_state:
                st.info(f"Selected: {st.session_state['fe_selected']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-footer">Powered by Hugging Face & Streamlit</div>', unsafe_allow_html=True)
    
    # If data is uploaded, process it
    if uploaded_file is not None and ('df' not in st.session_state or st.session_state.get('df') is None):
        try:
            # Attempt to read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Verify that we have a valid dataframe before storing in session state
            if df is not None and not df.empty:
                st.session_state['df'] = df
                st.session_state['descriptive_stats'] = df.describe()
                st.session_state['dataset_name'] = uploaded_file.name
                st.success(f"Successfully loaded dataset: {uploaded_file.name}")
            else:
                st.error("The uploaded file appears to be empty.")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Create navigation tabs using Streamlit
    st.write("### Navigation")
    tabs = ["Overview", "Distribution", "Relationships", "AI Insights", "Chat"]
    
    # Create columns for each tab
    cols = st.columns(len(tabs))
    
    # Handle tab selection using Streamlit buttons
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(tab, key=f"tab_{tab.lower()}"):
                st.session_state['selected_tab'] = f"tab-{tab.lower().replace(' ', '-')}"
                st.rerun()
    
    # Show selected tab indicator
    selected_tab_name = st.session_state['selected_tab'].replace('tab-', '').replace('-', ' ').title()
    st.markdown(f"<div style='text-align: center; margin-bottom: 2rem;'>Selected: {selected_tab_name}</div>", unsafe_allow_html=True)
    
    # Show welcome message if no data is uploaded
    if 'df' not in st.session_state:
        display_welcome_page()
    else:
        # Display content based on selected tab
        if st.session_state['selected_tab'] == 'tab-overview':
            display_descriptive_tab()
        elif st.session_state['selected_tab'] == 'tab-distribution':
            display_distribution_tab()
        elif st.session_state['selected_tab'] == 'tab-relationships':
            display_relationships_tab()
        elif st.session_state['selected_tab'] == 'tab-ai-insights' or st.session_state['selected_tab'] == 'tab-ai':
            display_ai_insights_tab()
        elif st.session_state['selected_tab'] == 'tab-chat':
            display_chat_interface()
    
    # After all tabs are rendered, check if we have a regenerate action
    # This is processed at the end to avoid session state changes during rendering
    if (st.session_state.get('loading_insights', False) and 
        ('ai_insights' not in st.session_state or st.session_state.get('ai_insights') is None)):
        logger.info("Generating AI insights at end of main function")
        try:
            st.session_state['ai_insights'] = generate_ai_insights()
            logger.info(f"Generated insights: {len(st.session_state['ai_insights'])} categories")
            st.session_state['loading_insights'] = False
        except Exception as e:
            logger.error(f"Error generating insights in main function: {str(e)}")
            st.session_state['loading_insights'] = False
            st.session_state['ai_insights'] = {}  # Set to empty dict to prevent repeated failures
        finally:
            st.rerun()

if __name__ == "__main__":
    main()