"""
Configuration constants for the Trade Policy Analysis project.
"""

from typing import List, Dict, Any

# Trade vocabulary for keyword detection
TRADE_VOCABULARY: List[str] = [
    "tariff",
    "import duty",
    "import barrier",
    "import ban",
    "import tax",
    "import subsidies",
    "export ban",
    "export tax",
    "export subsidies",
    "government subsidies",
    "GATT",
    "WTO",
    "World Trade Organization",
    "trade treaty",
    "trade agreement",
    "trade policy",
    "trade act",
    "trade relationship",
    "free trade",
    "Doha round",
    "Uruguay round",
    "dumping",
    "border tax",
]

# Event dates for trade policy analysis
EVENT_DATES: List[str] = [
    "2017-04-24",
    "2017-08-08",
    "2018-01-01",
    "2018-03-01",
    "2018-03-22",
    "2018-04-02",
    "2018-06-15",
    "2018-09-17",
    "2019-05-10",
    "2019-08-23",
    "2019-09-01",
    "2019-12-15",
    "2020-01-15",
    "2024-05-14",
    "2025-02-01",
    "2025-02-04",
    "2025-03-04",
    "2025-03-12",
    "2025-07-23",
]

# Data file paths
DATA_PATHS: Dict[str, str] = {
    "index_constituents": "data/RIY Index constituents.feather",
    "asset_returns": "data/RIY Index returns.feather",
    "market_returns": "data/total_return_russell.feather",
    "rf_returns": "data/us_daily_tpu_data.csv",
    "sentiment_dictionary": "data/Loughran-McDonald_MasterDictionary_1993-2024.csv",
    "transcripts_chunks": "data/transcripts/",  # Directory containing chunked transcript files
}

# Output paths
OUTPUT_PATHS: Dict[str, str] = {
    "plots": "outputs/descriptive_statistics_plots",
    "results": "outputs",
    "finbert_results": "outputs/finbert_trade_policy_results.csv",
}

# Analysis configuration
ANALYSIS_CONFIG: Dict[str, Any] = {
    "finbert_model": "ProsusAI/finbert",
    "sentiment_window": 10,
    "event_study": {
        "nb_periods_before_event_as_start_estimation_normal_returns": 252,
        "nb_periods_before_event_as_end_estimation_normal_returns": 7,
        "event_window": 7,
    },
}
