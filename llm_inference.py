"""
LLM Inference Module

This module handles all interactions with the Hugging Face API, allowing the application
to generate EDA insights and feature engineering recommendations from dataset analysis.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import Dict, List, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face client
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please add it to your .env file.")

try:
    hf_client = InferenceClient(token=HF_TOKEN)
    logger.info("Successfully initialized Hugging Face client")
except Exception as e:
    logger.error(f"Failed to initialize Hugging Face client: {str(e)}")
    raise

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

class LLMInference:
    """Class for interacting with LLM via Hugging Face API"""
    
    def __init__(self, model_id: str = MODEL_ID):
        """Initialize the LLM inference class with model ID"""
        self.model_id = model_id
        self.client = hf_client
        logger.info(f"LLMInference initialized with model: {model_id}")
    
    def generate_eda_insights(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate EDA insights based on dataset information
        
        Args:
            dataset_info (Dict): Dictionary containing dataset analysis
                - shape
                - columns
                - dtypes
                - missing_values
                - basic_stats
                - correlations
                - sample_data
        
        Returns:
            str: Detailed EDA insights and recommendations
        """
        prompt = self._construct_eda_prompt(dataset_info)
        logger.info("Generating EDA insights")
        
        try:
            start_time = time.time()
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1
            )
            elapsed_time = time.time() - start_time
            logger.info(f"EDA insights generated in {elapsed_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error generating EDA insights: {str(e)}")
            return f"Error generating EDA insights: {str(e)}"
    
    def generate_feature_engineering_recommendations(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate feature engineering recommendations based on dataset information
        
        Args:
            dataset_info (Dict): Dictionary containing dataset analysis
        
        Returns:
            str: Feature engineering recommendations
        """
        prompt = self._construct_feature_engineering_prompt(dataset_info)
        logger.info("Generating feature engineering recommendations")
        
        try:
            start_time = time.time()
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Feature engineering recommendations generated in {elapsed_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error generating feature engineering recommendations: {str(e)}")
            return f"Error generating feature engineering recommendations: {str(e)}"
    
    def generate_data_quality_insights(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate data quality insights based on dataset information
        
        Args:
            dataset_info (Dict): Dictionary containing dataset analysis
        
        Returns:
            str: Data quality insights and improvement recommendations
        """
        prompt = self._construct_data_quality_prompt(dataset_info)
        logger.info("Generating data quality insights")
        
        try:
            start_time = time.time()
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Data quality insights generated in {elapsed_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error generating data quality insights: {str(e)}")
            return f"Error generating data quality insights: {str(e)}"
    
    def answer_dataset_question(self, question: str, dataset_info: Dict[str, Any]) -> str:
        """
        Answer a specific question about the dataset
        
        Args:
            question (str): User's question about the dataset
            dataset_info (Dict): Dictionary containing dataset analysis
        
        Returns:
            str: Answer to the user's question
        """
        prompt = self._construct_qa_prompt(question, dataset_info)
        logger.info(f"Answering dataset question: {question[:50]}...")
        
        try:
            start_time = time.time()
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=512,
                temperature=0.4,
                top_p=0.95,
                repetition_penalty=1.1
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Answer generated in {elapsed_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error answering dataset question: {str(e)}")
            return f"Error answering dataset question: {str(e)}"
    
    def _construct_eda_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Construct a prompt for EDA insights"""
        
        # Extract key information from dataset_info
        shape = dataset_info.get("shape", "N/A")
        columns = dataset_info.get("columns", [])
        dtypes = dataset_info.get("dtypes", {})
        missing_values = dataset_info.get("missing_values", {})
        basic_stats = dataset_info.get("basic_stats", {})
        correlations = dataset_info.get("correlations", {})
        sample_data = dataset_info.get("sample_data", "N/A")
        
        # Format the information for the prompt
        columns_info = "\n".join([f"- {col} ({dtypes.get(col, 'unknown')})" for col in columns])
        missing_info = "\n".join([f"- {col}: {count} missing values ({percent}%)" 
                                 for col, (count, percent) in missing_values.items() if count > 0])
        
        if not missing_info:
            missing_info = "No missing values detected."
        
        # Construct the prompt
        prompt = f"""You are a data scientist tasked with performing Exploratory Data Analysis (EDA) on a dataset. 
Based on the following dataset information, provide comprehensive EDA insights:

Dataset Information:
- Shape: {shape}
- Columns and their types:
{columns_info}

- Missing values:
{missing_info}

- Basic statistics:
{basic_stats}

- Top correlations:
{correlations}

- Sample data:
{sample_data}

Please provide a detailed EDA analysis that includes:

1. Summary of the dataset (what it appears to be about, key features, etc.)
2. Distribution analysis of key variables
3. Relationship analysis between variables
4. Identification of patterns, outliers, or anomalies
5. Recommended visualizations that would be insightful
6. Initial hypotheses based on the data

Your analysis should be structured, thorough, and provide actionable insights for further investigation.
"""
        return prompt
    
    def _construct_feature_engineering_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Construct a prompt for feature engineering recommendations"""
        
        # Extract key information from dataset_info
        shape = dataset_info.get("shape", "N/A")
        columns = dataset_info.get("columns", [])
        dtypes = dataset_info.get("dtypes", {})
        basic_stats = dataset_info.get("basic_stats", {})
        correlations = dataset_info.get("correlations", {})
        
        # Format the information for the prompt
        columns_info = "\n".join([f"- {col} ({dtypes.get(col, 'unknown')})" for col in columns])
        
        # Construct the prompt
        prompt = f"""You are a machine learning engineer specializing in feature engineering. 
Based on the following dataset information, provide recommendations for feature engineering:

Dataset Information:
- Shape: {shape}
- Columns and their types:
{columns_info}

- Basic statistics:
{basic_stats}

- Top correlations:
{correlations}

Please provide comprehensive feature engineering recommendations that include:

1. Numerical feature transformations (scaling, normalization, log transforms, etc.)
2. Categorical feature encoding strategies
3. Feature interaction suggestions
4. Dimensionality reduction approaches if applicable
5. Time-based feature creation if applicable
6. Text processing techniques if there are text fields
7. Feature selection recommendations

For each recommendation, explain why it would be beneficial and how it could improve model performance.
Be specific to this dataset's characteristics rather than providing generic advice.
"""
        return prompt
    
    def _construct_data_quality_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Construct a prompt for data quality insights"""
        
        # Extract key information from dataset_info
        shape = dataset_info.get("shape", "N/A")
        columns = dataset_info.get("columns", [])
        dtypes = dataset_info.get("dtypes", {})
        missing_values = dataset_info.get("missing_values", {})
        basic_stats = dataset_info.get("basic_stats", {})
        
        # Format the information for the prompt
        columns_info = "\n".join([f"- {col} ({dtypes.get(col, 'unknown')})" for col in columns])
        missing_info = "\n".join([f"- {col}: {count} missing values ({percent}%)" 
                                 for col, (count, percent) in missing_values.items() if count > 0])
        
        if not missing_info:
            missing_info = "No missing values detected."
        
        # Construct the prompt
        prompt = f"""You are a data quality expert. 
Based on the following dataset information, provide data quality insights and recommendations:

Dataset Information:
- Shape: {shape}
- Columns and their types:
{columns_info}

- Missing values:
{missing_info}

- Basic statistics:
{basic_stats}

Please provide a comprehensive data quality assessment that includes:

1. Assessment of data completeness (missing values)
2. Identification of potential data inconsistencies or errors
3. Recommendations for data cleaning and preprocessing
4. Advice on handling outliers
5. Suggestions for data validation checks
6. Recommendations to improve data quality

Your assessment should be specific to this dataset and provide actionable recommendations.
"""
        return prompt
    
    def _construct_qa_prompt(self, question: str, dataset_info: Dict[str, Any]) -> str:
        """Construct a prompt for answering questions about the dataset"""
        
        # Extract key information from dataset_info
        shape = dataset_info.get("shape", "N/A")
        columns = dataset_info.get("columns", [])
        dtypes = dataset_info.get("dtypes", {})
        basic_stats = dataset_info.get("basic_stats", {})
        
        # Format the information for the prompt
        columns_info = "\n".join([f"- {col} ({dtypes.get(col, 'unknown')})" for col in columns])
        
        # Construct the prompt
        prompt = f"""You are a data scientist answering questions about a dataset. 
Based on the following dataset information, please answer the user's question:

Dataset Information:
- Shape: {shape}
- Columns and their types:
{columns_info}

- Basic statistics:
{basic_stats}

User's question: {question}

Please provide a clear, informative answer to the user's question based on the dataset information provided.
"""
        return prompt
