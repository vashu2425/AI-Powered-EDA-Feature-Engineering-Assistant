"""
EDA Analysis Module

This module handles all dataset processing and analysis, providing structured information
about the dataset that can be used for visualization and LLM prompting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64

class DatasetAnalyzer:
    """Class for analyzing datasets and extracting key information"""
    
    def __init__(self, df: pd.DataFrame = None):
        """Initialize with an optional dataframe"""
        self.df = df
        self.analysis_results = {}
    
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a dataframe for analysis"""
        self.df = df
        # Reset analysis results when loading a new dataframe
        self.analysis_results = {}
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the dataset
        
        Returns:
            Dict: Dictionary containing all analysis results
        """
        if self.df is None:
            raise ValueError("No dataframe loaded. Please load a dataframe first.")
        
        # Basic information
        self.analysis_results["shape"] = self.df.shape
        self.analysis_results["columns"] = list(self.df.columns)
        self.analysis_results["dtypes"] = {col: str(self.df[col].dtype) for col in self.df.columns}
        
        # Missing values
        self.analysis_results["missing_values"] = self._analyze_missing_values()
        
        # Basic statistics
        self.analysis_results["basic_stats"] = self._generate_basic_stats()
        
        # Correlations (for numerical columns)
        self.analysis_results["correlations"] = self._analyze_correlations()
        
        # Sample data
        self.analysis_results["sample_data"] = self.df.head().to_string()
        
        # Additional analyses
        self.analysis_results["categorical_columns"] = self._identify_categorical_columns()
        self.analysis_results["numerical_columns"] = self._identify_numerical_columns()
        self.analysis_results["unique_values"] = self._count_unique_values()
        
        return self.analysis_results
    
    def _analyze_missing_values(self) -> Dict[str, Tuple[int, float]]:
        """
        Analyze missing values in the dataset
        
        Returns:
            Dict: Column names as keys, tuples of (count, percentage) as values
        """
        missing_values = {}
        for col in self.df.columns:
            count = self.df[col].isna().sum()
            percentage = round((count / len(self.df)) * 100, 2)
            missing_values[col] = (count, percentage)
        
        return missing_values
    
    def _generate_basic_stats(self) -> str:
        """
        Generate basic statistics for the dataset
        
        Returns:
            str: String representation of basic statistics
        """
        # For numerical columns
        num_stats = self.df.describe().to_string()
        
        # For categorical columns
        cat_columns = self._identify_categorical_columns()
        cat_stats = ""
        if cat_columns:
            cat_stats = "\n\nCategorical columns statistics:\n"
            for col in cat_columns:
                value_counts = self.df[col].value_counts().head(10)
                cat_stats += f"\n{col} - Top values:\n{value_counts.to_string()}\n"
        
        return num_stats + cat_stats
    
    def _analyze_correlations(self) -> str:
        """
        Analyze correlations between numerical features
        
        Returns:
            str: String representation of top correlations
        """
        num_columns = self._identify_numerical_columns()
        
        if not num_columns or len(num_columns) < 2:
            return "Not enough numerical columns for correlation analysis."
        
        corr_matrix = self.df[num_columns].corr()
        
        # Get top correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(num_columns)):
            for j in range(i+1, len(num_columns)):
                col1, col2 = num_columns[i], num_columns[j]
                corr_value = corr_matrix.loc[col1, col2]
                if not np.isnan(corr_value):
                    corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Format results
        result = "Top correlations:\n"
        for col1, col2, corr in corr_pairs[:10]:  # Top 10 correlations
            result += f"{col1} -- {col2}: {corr:.4f}\n"
        
        return result
    
    def _identify_categorical_columns(self) -> List[str]:
        """
        Identify categorical columns in the dataset
        
        Returns:
            List[str]: List of categorical column names
        """
        cat_columns = []
        for col in self.df.columns:
            # Consider object, category, and boolean types as categorical
            if self.df[col].dtype == 'object' or self.df[col].dtype == 'category' or self.df[col].dtype == 'bool':
                cat_columns.append(col)
            # Also consider int/float columns with few unique values as categorical
            elif (self.df[col].dtype == 'int64' or self.df[col].dtype == 'float64') and \
                 self.df[col].nunique() < 10 and self.df[col].nunique() / len(self.df) < 0.05:
                cat_columns.append(col)
        
        return cat_columns
    
    def _identify_numerical_columns(self) -> List[str]:
        """
        Identify numerical columns in the dataset
        
        Returns:
            List[str]: List of numerical column names
        """
        num_columns = []
        cat_columns = self._identify_categorical_columns()
        
        for col in self.df.columns:
            if col not in cat_columns and pd.api.types.is_numeric_dtype(self.df[col].dtype):
                num_columns.append(col)
        
        return num_columns
    
    def _count_unique_values(self) -> Dict[str, int]:
        """
        Count unique values for each column
        
        Returns:
            Dict: Column names as keys, unique count as values
        """
        return {col: self.df[col].nunique() for col in self.df.columns}
    
    def generate_eda_visualizations(self) -> Dict[str, str]:
        """
        Generate common EDA visualizations
        
        Returns:
            Dict: Dictionary of visualization titles and their base64-encoded images
        """
        if self.df is None:
            raise ValueError("No dataframe loaded. Please load a dataframe first.")
        
        visualizations = {}
        
        # 1. Missing values heatmap
        visualizations["missing_values_heatmap"] = self._plot_missing_values()
        
        # 2. Distribution plots for numerical features
        num_columns = self._identify_numerical_columns()
        for i, col in enumerate(num_columns[:5]):  # Limit to first 5 numerical columns
            visualizations[f"distribution_{col}"] = self._plot_distribution(col)
        
        # 3. Correlation heatmap
        visualizations["correlation_heatmap"] = self._plot_correlation_heatmap()
        
        # 4. Categorical feature distributions
        cat_columns = self._identify_categorical_columns()
        for i, col in enumerate(cat_columns[:5]):  # Limit to first 5 categorical columns
            visualizations[f"categorical_{col}"] = self._plot_categorical_distribution(col)
        
        # 5. Scatter plot of 2 most correlated features
        if len(num_columns) >= 2:
            visualizations["scatter_plot"] = self._plot_scatter_correlation()
        
        return visualizations
    
    def _plot_missing_values(self) -> str:
        """Generate missing values heatmap"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cmap='viridis', yticklabels=False, cbar=True, cbar_kws={'label': 'Missing Data'})
        plt.tight_layout()
        plt.title('Missing Values Heatmap')
        
        # Convert plot to base64 string
        return self._fig_to_base64(plt.gcf())
    
    def _plot_distribution(self, column: str) -> str:
        """Generate distribution plot for a numerical column"""
        plt.figure(figsize=(10, 6))
        
        # Histogram with KDE
        sns.histplot(data=self.df, x=column, kde=True)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Convert plot to base64 string
        return self._fig_to_base64(plt.gcf())
    
    def _plot_correlation_heatmap(self) -> str:
        """Generate correlation heatmap"""
        num_columns = self._identify_numerical_columns()
        
        if not num_columns or len(num_columns) < 2:
            return ""
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[num_columns].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Custom diverging palette
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Convert plot to base64 string
        return self._fig_to_base64(plt.gcf())
    
    def _plot_categorical_distribution(self, column: str) -> str:
        """Generate bar plot for categorical column"""
        plt.figure(figsize=(10, 6))
        
        # Get value counts and limit to top 10 categories if there are too many
        value_counts = self.df[column].value_counts()
        if len(value_counts) > 10:
            # Keep top 9 categories and group the rest as 'Other'
            top_categories = value_counts.nlargest(9).index
            data = self.df.copy()
            data[column] = data[column].apply(lambda x: x if x in top_categories else 'Other')
            sns.countplot(y=column, data=data, order=data[column].value_counts().index)
        else:
            sns.countplot(y=column, data=self.df, order=value_counts.index)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel('Count')
        plt.ylabel(column)
        plt.tight_layout()
        
        # Convert plot to base64 string
        return self._fig_to_base64(plt.gcf())
    
    def _plot_scatter_correlation(self) -> str:
        """Generate scatter plot of two most correlated features"""
        num_columns = self._identify_numerical_columns()
        
        if not num_columns or len(num_columns) < 2:
            return ""
        
        # Find the two most correlated features
        corr_matrix = self.df[num_columns].corr().abs()
        
        # Get upper triangle mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix = corr_matrix.mask(mask)
        
        # Find the max correlation
        max_corr = corr_matrix.max().max()
        max_corr_idx = corr_matrix.stack().idxmax()
        
        if pd.isna(max_corr):
            return ""
        
        # Get the column names
        col1, col2 = max_corr_idx
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Add regression line
        sns.regplot(x=col1, y=col2, data=self.df, scatter_kws={'alpha': 0.5})
        
        plt.title(f'Scatter plot of {col1} vs {col2} (correlation: {corr_matrix.loc[col1, col2]:.2f})')
        plt.tight_layout()
        
        # Convert plot to base64 string
        return self._fig_to_base64(plt.gcf())
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    def suggest_data_preprocessing(self) -> Dict[str, List[str]]:
        """
        Suggest preprocessing steps based on dataset analysis
        
        Returns:
            Dict: Dictionary of preprocessing suggestions for each column type
        """
        if not self.analysis_results:
            self.analyze_dataset()
        
        suggestions = {
            "numerical": [],
            "categorical": [],
            "missing_values": [],
            "outliers": [],
            "general": []
        }
        
        # Missing values suggestions
        missing_cols = [col for col, (count, _) in self.analysis_results["missing_values"].items() if count > 0]
        if missing_cols:
            suggestions["missing_values"].append(f"Found {len(missing_cols)} columns with missing values.")
            if len(missing_cols) > 5:
                suggestions["missing_values"].append(f"Columns with highest missing values: {', '.join(missing_cols[:5])}...")
            else:
                suggestions["missing_values"].append(f"Columns with missing values: {', '.join(missing_cols)}")
            
            suggestions["missing_values"].append("Consider these strategies for handling missing values:")
            suggestions["missing_values"].append("- Imputation (mean/median for numerical, mode for categorical)")
            suggestions["missing_values"].append("- Creating missing value indicators as new features")
            suggestions["missing_values"].append("- Removing rows or columns with too many missing values")
        
        # Numerical column suggestions
        num_cols = self.analysis_results["numerical_columns"]
        if num_cols:
            suggestions["numerical"].append(f"Found {len(num_cols)} numerical columns.")
            suggestions["numerical"].append("Consider these preprocessing steps:")
            suggestions["numerical"].append("- Scaling (StandardScaler or MinMaxScaler)")
            suggestions["numerical"].append("- Check for skewness and apply log or Box-Cox transformation if needed")
            suggestions["numerical"].append("- Create binned versions of continuous variables")
            
            # Check for potential outliers
            for col in num_cols:
                if col in self.df.columns:  # Safety check
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    iqr = q3 - q1
                    outlier_count = ((self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))).sum()
                    
                    if outlier_count > 0:
                        percentage = round((outlier_count / len(self.df)) * 100, 2)
                        if percentage > 5:  # If more than 5% are outliers
                            suggestions["outliers"].append(f"Column '{col}' has {outlier_count} potential outliers ({percentage}%).")
        
        # Categorical column suggestions
        cat_cols = self.analysis_results["categorical_columns"]
        if cat_cols:
            suggestions["categorical"].append(f"Found {len(cat_cols)} categorical columns.")
            
            # Check cardinality (number of unique values)
            high_cardinality = []
            for col in cat_cols:
                unique_count = self.analysis_results["unique_values"].get(col, 0)
                if unique_count > 10:
                    high_cardinality.append((col, unique_count))
            
            if high_cardinality:
                suggestions["categorical"].append("High cardinality columns (many unique values):")
                for col, count in sorted(high_cardinality, key=lambda x: x[1], reverse=True)[:5]:
                    suggestions["categorical"].append(f"- {col}: {count} unique values")
                
                suggestions["categorical"].append("For high cardinality columns, consider:")
                suggestions["categorical"].append("- Grouping less frequent categories")
                suggestions["categorical"].append("- Target encoding or embedding techniques")
            
            suggestions["categorical"].append("General categorical encoding strategies:")
            suggestions["categorical"].append("- One-hot encoding for low cardinality columns")
            suggestions["categorical"].append("- Label encoding for ordinal variables")
        
        # General suggestions
        suggestions["general"].append("General preprocessing recommendations:")
        suggestions["general"].append("- Check for duplicate rows and remove if necessary")
        suggestions["general"].append("- Normalize text fields (lowercase, remove special characters)")
        suggestions["general"].append("- Create feature interactions for highly correlated features")
        
        return suggestions
        
    def generate_feature_engineering_ideas(self) -> List[str]:
        """
        Generate feature engineering ideas based on dataset analysis
        
        Returns:
            List[str]: List of feature engineering suggestions
        """
        if not self.analysis_results:
            self.analyze_dataset()
        
        ideas = []
        
        # Get column types
        num_cols = self.analysis_results["numerical_columns"]
        cat_cols = self.analysis_results["categorical_columns"]
        
        # Aggregation features
        if len(num_cols) >= 2:
            ideas.append("### Numerical Feature Transformations:")
            ideas.append("1. Create polynomial features for continuous variables")
            ideas.append("2. Apply mathematical transformations (log, sqrt, square) to handle skewed distributions")
            ideas.append("3. Create binned versions of continuous features to capture non-linear relationships")
            
            # Check for date/time related column names
            time_related_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['date', 'time', 'year', 'month', 'day'])]
            if time_related_cols:
                ideas.append("\n### Time-Based Features:")
                ideas.append(f"Detected potential date/time columns: {', '.join(time_related_cols)}")
                ideas.append("1. Extract components like year, month, day, weekday, quarter")
                ideas.append("2. Create cyclical features using sine/cosine transformations for periodic time components")
                ideas.append("3. Calculate time since specific events or time differences between dates")
        
        # Categorical interactions
        if len(cat_cols) >= 2:
            ideas.append("\n### Categorical Feature Engineering:")
            ideas.append("1. Create interaction features by combining categorical variables")
            ideas.append("2. Use target encoding for high cardinality categorical features")
            ideas.append("3. Combine rare categories into an 'Other' category to reduce dimensionality")
        
        # Mixed interactions
        if num_cols and cat_cols:
            ideas.append("\n### Feature Interactions:")
            ideas.append("1. Create group-based statistics (mean, median, min, max) of numerical features grouped by categorical features")
            ideas.append("2. Calculate the difference from group means for numerical features")
            ideas.append("3. Create ratio or difference features between related numerical columns")
        
        # Dimensionality reduction
        if len(num_cols) > 10:
            ideas.append("\n### Dimensionality Reduction:")
            ideas.append("1. Apply PCA to reduce dimensionality and create principal components")
            ideas.append("2. Use feature selection methods (information gain, chi-square, mutual information)")
            ideas.append("3. Try UMAP or t-SNE for non-linear dimensionality reduction")
        
        # Text features
        text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object' and
                     self.df[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 3).mean() > 0.5]
        if text_cols:
            ideas.append("\n### Text Feature Engineering:")
            ideas.append(f"Detected potential text columns: {', '.join(text_cols)}")
            ideas.append("1. Create bag-of-words or TF-IDF representations")
            ideas.append("2. Extract text length, word count, and other statistical features")
            ideas.append("3. Consider pretrained word embeddings or sentence transformers")
        
        return ideas
