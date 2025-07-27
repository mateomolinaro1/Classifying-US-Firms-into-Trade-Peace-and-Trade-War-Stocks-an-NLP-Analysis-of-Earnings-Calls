# NLP Trade Policy Sensitivity Analysis

A comprehensive Natural Language Processing pipeline for analyzing the sensitivity of stocks to trade policy announcements using earnings call transcripts.

## Overview

This project analyzes how companies' earnings call transcripts reveal their exposure to trade policy shocks. It uses multiple NLP approaches including:

- **Bag of Words (BOW)** analysis with trade-specific vocabulary
- **Bag of Words with Sentiment (BOWWS)** analysis
- **FinBERT** transformer model for advanced sentiment analysis
- Statistical analysis and visualization of results

## Key Features

- 📊 **Multi-dimensional Analysis**: BOW, sentiment-enhanced BOW, and FinBERT analysis
- 🎯 **Trade-Specific Vocabulary**: Curated list of trade policy keywords
- 📈 **Event Study Framework**: Analysis around key trade policy announcement dates
- 🔄 **Automated Pipeline**: End-to-end orchestrated analysis workflow
- 📊 **Rich Visualizations**: Comprehensive plots and statistical summaries
- ⚡ **GPU Support**: CUDA acceleration for FinBERT analysis

## Requirements

- Python 3.11-3.12
- Poetry (for dependency management)
- CUDA-compatible GPU (optional, for faster FinBERT processing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/enzomontariol/nlp_tariffs_sensitivity_of_stocks.git
cd nlp_tariffs_sensitivity_of_stocks
```

### 2. Install Poetry
If you don't have Poetry installed, run:
```bash
pipx install poetry
```

or follow the [Poetry installation guide](https://python-poetry.org/docs/#installation).

### 3. Install Dependencies with Poetry
```bash
# Install project dependencies
poetry install --no-root

# Activate the virtual environment
poetry shell
```

### Alternative: Manual Installation
If you prefer pip:
```bash
pip install -r requirements.txt  # You'll need to generate this from pyproject.toml
```

## Data Setup

The project expects the following data files in the `data/` directory:

- `RIY Index constituents.feather` - Index constituent data
- `RIY Index returns.feather` - Asset returns data  
- `total_return_russell.feather` - Market returns data
- `us_daily_tpu_data.csv` - Risk-free rate data
- `Loughran-McDonald_MasterDictionary_1993-2024.csv` - Sentiment dictionary
- `transcripts/` - Directory containing chunked transcript files (`formatted_transcripts_gzip_chunk_*.pkl`)

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
# Using Poetry (recommended)
poetry run python main.py

# Or if environment is activated
python main.py
```

### Configuration

The analysis can be configured in `src/config.py`:

- **Trade Vocabulary**: Customize trade policy keywords
- **Event Dates**: Modify trade policy announcement dates
- **File Paths**: Update data file locations
- **Analysis Settings**: Adjust analysis parameters

### Analysis Components

#### 1. Bag of Words Analysis
Counts occurrences of trade policy keywords in transcripts:
```python
from src.nlp_models import BagOfWords
bow = BagOfWords(transcript_data, vocabulary=TRADE_VOCABULARY)
bow.create_bag_of_words_for_transcripts()
```

#### 2. Sentiment-Enhanced Analysis
Combines keyword analysis with sentiment scoring:
```python
from src.nlp_models import BagOfWordsWithSentiment
bowws = BagOfWordsWithSentiment(transcript_data, sentiment_dict_path)
```

#### 3. FinBERT Analysis
Advanced transformer-based sentiment analysis:
```python
from src.transform import TradePolicyShockSentimentAnalyzer
analyzer = TradePolicyShockSentimentAnalyzer()
results = analyzer.analyze_dataframe(transcript_data)
```

### Custom Analysis

For custom analysis, use the orchestrator:

```python
from src.analysis_orchestrator import TradePolicyAnalyzer

# Initialize analyzer
analyzer = TradePolicyAnalyzer(
    output_dir="custom_output",
    run_finbert=True,
    event_dates=["2018-03-01", "2018-06-15"]
)

# Run analysis
results = analyzer.run_complete_analysis()
summary = analyzer.get_results_summary()
```

## Output

The analysis generates:

### Results Files
- `outputs/descriptive_statistics_plots/` - Visualization plots
- `outputs/finbert_trade_policy_results.csv` - FinBERT analysis results
- Analysis logs and summary statistics

### Visualizations
- Daily and monthly mean frequency plots
- Event-based analysis charts
- Sentiment distribution plots
- Statistical summary visualizations

## Project Structure

```
nlp_tariffs_sensitivity_of_stocks/
├── main.py                     # Main execution script
├── pyproject.toml             # Poetry configuration
├── data/                      # Data files
│   ├── transcripts/          # Chunked transcript files
│   └── *.feather, *.csv      # Market and sentiment data
├── src/                      # Source code
│   ├── analysis_orchestrator.py  # Main analysis coordinator
│   ├── config.py             # Configuration constants
│   ├── nlp_models.py         # BOW and sentiment models
│   ├── transform.py          # FinBERT analysis
│   ├── data_loader.py        # Data loading utilities
│   ├── descriptive_stats.py  # Statistical analysis
│   ├── visualization.py      # Plotting functions
│   ├── statistical_analysis.py # Event study analysis
│   └── utilities.py          # Helper functions
├── outputs/                  # Generated results
└── docs/                     # Documentation
```

## Performance Notes

- **GPU Acceleration**: FinBERT analysis will automatically use CUDA if available
- **Memory Usage**: Large transcript datasets are processed in chunks
- **Processing Time**: Complete analysis of 54K+ transcripts takes ~4-5 minutes with GPU (tested on NVIDIA RTX 4070)

## Sample Output

```
============================================================
ANALYSIS COMPLETE - SUMMARY:
============================================================
data_loaded: True
bow_completed: True  
bowws_completed: True
finbert_completed: True
total_transcripts: 54851
bow_results_shape: (54851, 1)
bowws_results_shape: (54851, 1) 
finbert_results_shape: (54851, 9)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in FinBERT analysis
2. **Missing Data Files**: Ensure all required data files are in `data/` directory
3. **Poetry Installation**: Use `poetry install --no-root` if package installation fails

### GPU Setup
For CUDA support:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch if needed
poetry add torch --source https://download.pytorch.org/whl/cu129
```
## Authors
- Matéo Molinaro : mateo0609@hotmail.fr
- Enzo Montariol : enzo.montariol@gmail.com