"""
LLM Inference Module

This module handles all interactions with the Groq API via LangChain,
allowing the application to generate EDA insights and feature engineering
recommendations from dataset analysis.
"""

import os
from dotenv import load_dotenv
import logging
import time
from typing import Dict, Any, List, Optional
from langchain_community.callbacks.manager import get_openai_callback

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnableSequence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

# Create LLM model
try:
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    logger.info("Successfully initialized Groq client")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

class LLMInference:
    """Class for interacting with LLM via Groq API using LangChain"""
    
    def __init__(self, model_id: str = "llama3-8b-8192"):
        """Initialize the LLM inference class with Groq model"""
        self.model_id = model_id
        self.llm = llm
        
        # Initialize prompt templates and chains
        self._init_prompt_templates()
        self._init_chains()
        
        logger.info(f"LLMInference initialized with model: {model_id}")
    
    def _init_prompt_templates(self):
        """Initialize all prompt templates"""
        
        # EDA insights prompt template
        self.eda_prompt_template = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """You are a data scientist tasked with performing Exploratory Data Analysis (EDA) on a dataset. 
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
            )
        ])
        
        # Feature engineering prompt template
        self.feature_engineering_prompt_template = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """You are a machine learning engineer specializing in feature engineering. 
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
            )
        ])
        
        # Data quality prompt template
        self.data_quality_prompt_template = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """You are a data quality expert. 
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
            )
        ])
        
        # QA prompt template
        self.qa_prompt_template = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """You are a data scientist answering questions about a dataset. 
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
            )
        ])
    
    def _init_chains(self):
        """Initialize all chains using modern RunnableSequence pattern"""
        
        # EDA insights chain
        self.eda_chain = self.eda_prompt_template | self.llm
        
        # Feature engineering chain
        self.feature_engineering_chain = self.feature_engineering_prompt_template | self.llm
        
        # Data quality chain
        self.data_quality_chain = self.data_quality_prompt_template | self.llm
        
        # QA chain
        self.qa_chain = self.qa_prompt_template | self.llm
    
    def _format_columns_info(self, columns: List[str], dtypes: Dict[str, str]) -> str:
        """Format columns info for prompt"""
        return "\n".join([f"- {col} ({dtypes.get(col, 'unknown')})" for col in columns])
    
    def _format_missing_info(self, missing_values: Dict[str, tuple]) -> str:
        """Format missing values info for prompt"""
        missing_info = "\n".join([f"- {col}: {count} missing values ({percent}%)" 
                               for col, (count, percent) in missing_values.items() if count > 0])
        
        if not missing_info:
            missing_info = "No missing values detected."
            
        return missing_info
    
    def _execute_chain(
        self, 
        chain: RunnableSequence, 
        input_data: Dict[str, Any], 
        operation_name: str
    ) -> str:
        """
        Execute a chain with tracking and error handling
        
        Args:
            chain: The LangChain chain to execute
            input_data: The input data for the chain
            operation_name: Name of the operation for logging
            
        Returns:
            str: The generated text
        """
        try:
            start_time = time.time()
            with get_openai_callback() as cb:
                result = chain.invoke(input_data).content
            elapsed_time = time.time() - start_time
            
            logger.info(f"{operation_name} generated in {elapsed_time:.2f} seconds")
            logger.info(f"Tokens used: {cb.total_tokens}, "
                      f"Prompt tokens: {cb.prompt_tokens}, "
                      f"Completion tokens: {cb.completion_tokens}")
            
            return result
        except Exception as e:
            error_msg = f"Error executing {operation_name.lower()}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def generate_eda_insights(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate EDA insights based on dataset information using LangChain
        
        Args:
            dataset_info: Dictionary containing dataset analysis
        
        Returns:
            str: Detailed EDA insights and recommendations
        """
        logger.info("Generating EDA insights")
        
        # Format the input data
        columns_info = self._format_columns_info(
            dataset_info.get("columns", []), 
            dataset_info.get("dtypes", {})
        )
        
        missing_info = self._format_missing_info(
            dataset_info.get("missing_values", {})
        )
        
        # Prepare input for the chain
        input_data = {
            "shape": dataset_info.get("shape", "N/A"),
            "columns_info": columns_info,
            "missing_info": missing_info,
            "basic_stats": dataset_info.get("basic_stats", ""),
            "correlations": dataset_info.get("correlations", ""),
            "sample_data": dataset_info.get("sample_data", "N/A")
        }
        
        return self._execute_chain(self.eda_chain, input_data, "EDA insights")
    
    def generate_feature_engineering_recommendations(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate feature engineering recommendations based on dataset information using LangChain
        
        Args:
            dataset_info: Dictionary containing dataset analysis
        
        Returns:
            str: Feature engineering recommendations
        """
        logger.info("Generating feature engineering recommendations")
        
        # Format the input data
        columns_info = self._format_columns_info(
            dataset_info.get("columns", []), 
            dataset_info.get("dtypes", {})
        )
        
        # Prepare input for the chain
        input_data = {
            "shape": dataset_info.get("shape", "N/A"),
            "columns_info": columns_info,
            "basic_stats": dataset_info.get("basic_stats", ""),
            "correlations": dataset_info.get("correlations", "")
        }
        
        return self._execute_chain(
            self.feature_engineering_chain, 
            input_data, 
            "Feature engineering recommendations"
        )
    
    def generate_data_quality_insights(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate data quality insights based on dataset information using LangChain
        
        Args:
            dataset_info: Dictionary containing dataset analysis
        
        Returns:
            str: Data quality insights and improvement recommendations
        """
        logger.info("Generating data quality insights")
        
        # Format the input data
        columns_info = self._format_columns_info(
            dataset_info.get("columns", []), 
            dataset_info.get("dtypes", {})
        )
        
        missing_info = self._format_missing_info(
            dataset_info.get("missing_values", {})
        )
        
        # Prepare input for the chain
        input_data = {
            "shape": dataset_info.get("shape", "N/A"),
            "columns_info": columns_info,
            "missing_info": missing_info,
            "basic_stats": dataset_info.get("basic_stats", "")
        }
        
        return self._execute_chain(
            self.data_quality_chain, 
            input_data, 
            "Data quality insights"
        )
    
    def answer_dataset_question(self, question: str, dataset_info: Dict[str, Any]) -> str:
        """
        Answer a specific question about the dataset using LangChain
        
        Args:
            question: User's question about the dataset
            dataset_info: Dictionary containing dataset analysis
        
        Returns:
            str: Answer to the user's question
        """
        logger.info(f"Answering dataset question: {question[:50]}...")
        
        # Format the input data
        columns_info = self._format_columns_info(
            dataset_info.get("columns", []), 
            dataset_info.get("dtypes", {})
        )
        
        # Prepare input for the chain
        input_data = {
            "shape": dataset_info.get("shape", "N/A"),
            "columns_info": columns_info,
            "basic_stats": dataset_info.get("basic_stats", ""),
            "question": question
        }
        
        return self._execute_chain(
            self.qa_chain, 
            input_data, 
            "Answer"
        )
        
    def answer_with_memory(self, question: str, dataset_info: Dict[str, Any], memory) -> str:
        """
        Answer a question with conversation memory to maintain context
        
        Args:
            question: User's question about the dataset
            dataset_info: Dictionary containing dataset analysis
            memory: ConversationBufferMemory instance to store conversation history
            
        Returns:
            str: Answer to the user's question with conversation context
        """
        logger.info(f"Answering with memory: {question[:50]}...")
        
        # Format the input data for the dataset context
        columns_info = self._format_columns_info(
            dataset_info.get("columns", []), 
            dataset_info.get("dtypes", {})
        )
        
        # Create a custom prompt that includes both conversation history and dataset info
        memory_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """You are a data scientist answering questions about a dataset. 
The following is information about the dataset:

Dataset Information:
- Shape: {shape}
- Columns and their types:
{columns_info}

- Basic statistics:
{basic_stats}

Previous conversation:
{chat_history}

User's new question: {question}

Please provide a clear, informative answer to the user's question. Take into account the previous conversation for context. Make your answer specific to the dataset information provided."""
            )
        ])
        
        # Create a chain that uses both the prompt and memory
        memory_chain = memory_prompt | self.llm
        
        # Prepare the input data including memory retrieved from conversation_memory
        try:
            chat_history = memory.load_memory_variables({})["chat_history"]
            # Format chat history into a string
            chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        except Exception as e:
            logger.warning(f"Error loading memory: {str(e)}. Using empty chat history.")
            chat_history_str = "No previous conversation."
            
        input_data = {
            "shape": dataset_info.get("shape", "N/A"),
            "columns_info": columns_info,
            "basic_stats": dataset_info.get("basic_stats", ""),
            "question": question,
            "chat_history": chat_history_str
        }
        
        # Execute the chain and get a response
        response = self._execute_chain(
            memory_chain, 
            input_data, 
            "Answer with memory"
        )
        
        # Save the interaction to memory
        memory.save_context(
            {"input": question},
            {"output": response}
        )
        
        return response
