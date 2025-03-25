# AI-Powered EDA & Feature Engineering Assistant

![App Banner](https://raw.githubusercontent.com/vashu2425/AI-Powered-EDA-Feature-Engineering-Assistant/main/assets/banner.png)

An interactive application that uses AI to analyze datasets and provide comprehensive exploratory data analysis (EDA) insights and feature engineering recommendations.

## 🌟 Features

- **🤖 AI-Powered Analysis**: Receive detailed EDA insights generated by Mistral-7B
- **📊 Automated Visualizations**: Generate key visualizations with a single click
- **🔧 Feature Engineering Recommendations**: Get AI suggestions for improving your data
- **⚠️ Data Quality Assessment**: Identify issues in your dataset and receive fixing advice 
- **💬 Chat Interface**: Ask questions about your dataset and get AI-powered answers
- **🌙 Dark Mode UI**: Sleek, modern dark interface for comfortable analysis

## 📋 Demo

Here's a quick look at what you can do:

1. Upload a CSV dataset
2. Get automatic visualizations and statistics
3. Generate AI-powered insights for:
   - Exploratory Data Analysis
   - Feature Engineering Recommendations
   - Data Quality Assessment
4. Chat with your data to ask specific questions

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, Matplotlib, Seaborn
- **AI Integration**: LangChain + Hugging Face API
- **LLM Model**: Mistral-7B-Instruct-v0.3

## 📦 Installation

### Prerequisites
- Python 3.8+
- Anaconda or Miniconda (recommended)
- Hugging Face API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vashu2425/AI-Powered-EDA-Feature-Engineering-Assistant.git
cd AI-Powered-EDA-Feature-Engineering-Assistant
```

2. Create and activate a conda environment:
```bash
conda create -n ai_eda_env python=3.10
conda activate ai_eda_env
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Hugging Face API key:
```
HF_TOKEN=your_huggingface_token_here
```

## 🚀 Usage

1. Activate the conda environment:
```bash
conda activate ai_eda_env
```

2. Run the application:
```bash
streamlit run main.py
```

3. Open your web browser and navigate to `http://localhost:8501`

4. Upload a CSV dataset and start exploring!

## 📊 Example Analysis

Here are some examples of insights you can get:

- Comprehensive EDA insights about your dataset variables and distributions
- Feature engineering ideas specific to your data
- Data quality improvement recommendations
- Visualizations including correlation heatmaps, distribution plots, and more

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For any questions or feedback, please reach out to the repository owner.

---

### 🌟 Star this repository if you find it useful!
