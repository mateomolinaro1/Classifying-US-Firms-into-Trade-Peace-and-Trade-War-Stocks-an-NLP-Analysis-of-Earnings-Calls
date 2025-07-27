import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from .transform import TradePolicyShockSentimentAnalyzer

# Trade vocabulary constants
TRADE_VOCABULARY = [
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


class BagOfWords:
    """Bag of Words model for trade policy keyword analysis."""

    def __init__(
        self,
        formatted_transcripts_preprocessed: pd.DataFrame,
        vocabulary: Optional[List[str]] = None,
    ):
        if vocabulary is None:
            vocabulary = TRADE_VOCABULARY
        self.formatted_transcripts_preprocessed = formatted_transcripts_preprocessed
        self.vocabulary = vocabulary
        self.bag_of_words_frequency_by_words = None
        self.bag_of_words_frequency_sum = None

    @staticmethod
    def create_bag_of_words(
        text: str, vocabulary: List[str], frequency: bool = True
    ) -> Dict[str, float]:
        """Create bag of words representation for a single transcript."""
        if pd.isna(text) or not text:
            return {word: 0.0 for word in vocabulary}

        words = text.split()
        bag = {word: 0.0 for word in vocabulary}

        for word in words:
            if word in bag:
                bag[word] += 1.0 if not frequency else 1.0 / len(words)

        return bag

    def create_bag_of_words_for_transcripts(self):
        """Create bag of words representation for all transcripts."""
        bag_of_words = pd.DataFrame()
        bag_of_words["bag_of_words"] = self.formatted_transcripts_preprocessed[
            "transcript"
        ].apply(lambda x: self.create_bag_of_words(x, self.vocabulary, frequency=True))
        bag_of_words = bag_of_words["bag_of_words"].apply(pd.Series)
        bag_of_words["sum"] = bag_of_words.sum(axis=1)

        self.bag_of_words_frequency_by_words = bag_of_words.drop(columns=["sum"])
        self.bag_of_words_frequency_sum = bag_of_words[["sum"]]
        return


class DataSentimentDictionary:
    """Class to handle sentiment dictionary data."""

    def __init__(
        self,
        sentiment_dictionary_path: str = "data/Loughran-McDonald_MasterDictionary_1993-2024.csv",
    ):
        self.sentiment_dictionary_path = sentiment_dictionary_path
        self.positive_sentiment_dictionary = None
        self.negative_sentiment_dictionary = None
        self.load_sentiment_dictionary()

    def load_sentiment_dictionary(self):
        """Load sentiment dictionary from CSV file."""
        try:
            df = pd.read_csv(self.sentiment_dictionary_path)
            self.positive_sentiment_dictionary = list(
                df[df["Positive"] != 0]["Word"].str.lower().tolist()
            )
            self.negative_sentiment_dictionary = list(
                df[df["Negative"] != 0]["Word"].str.lower().tolist()
            )
        except Exception as e:
            print(f"Error loading sentiment dictionary: {e}")
            self.positive_sentiment_dictionary = []
            self.negative_sentiment_dictionary = []


class BagOfWordsWithSentiment(BagOfWords):
    """Bag of Words model with sentiment analysis."""

    def __init__(
        self,
        formatted_transcripts_preprocessed: pd.DataFrame,
        vocabulary: Optional[List[str]] = None,
        sentiment_dictionary_path: str = "data/Loughran-McDonald_MasterDictionary_1993-2024.csv",
        window: int = 10,
    ):

        super().__init__(formatted_transcripts_preprocessed, vocabulary)
        self.sentiment_dictionary_path = sentiment_dictionary_path
        self.window = window

        dsd = DataSentimentDictionary(
            sentiment_dictionary_path=self.sentiment_dictionary_path
        )
        self.positive_sentiment_dictionary = dsd.positive_sentiment_dictionary or []
        self.negative_sentiment_dictionary = dsd.negative_sentiment_dictionary or []
        self.bag_of_words_with_sentiment = None

    @staticmethod
    def create_bag_of_words_with_sentiment(
        text: str,
        trade_keywords: List[str],
        pos_words: List[str],
        neg_words: List[str],
        window: int = 10,
    ) -> float:
        """Compute trade sentiment using Hassan et al. (2019) methodology."""
        if pd.isna(text) or not text:
            return 0.0

        words = text.split()
        n = len(words)
        if n == 0:
            return 0.0

        sentiment_sum = 0

        for i, word in enumerate(words):
            if word.lower() in trade_keywords:
                window_start = max(0, i - window)
                window_end = min(n, i + window + 1)
                local_window = words[window_start:window_end]

                local_sentiment = 0
                for c in local_window:
                    if c.lower() in pos_words:
                        local_sentiment += 1
                    elif c.lower() in neg_words:
                        local_sentiment -= 1

                sentiment_sum += local_sentiment

        trade_sentiment = sentiment_sum / n if n > 0 else 0
        return trade_sentiment

    def create_bag_of_words_with_sentiment_for_transcripts(self):
        """Create sentiment-weighted bag of words for all transcripts."""
        bag_of_words_with_sentiment = pd.DataFrame()
        bag_of_words_with_sentiment[
            "bag_of_words_with_sentiment"
        ] = self.formatted_transcripts_preprocessed["transcript"].apply(
            lambda x: self.create_bag_of_words_with_sentiment(
                text=x,
                trade_keywords=self.vocabulary,
                pos_words=self.positive_sentiment_dictionary,
                neg_words=self.negative_sentiment_dictionary,
                window=self.window,
            )
        )

        bag_of_words_with_sentiment = bag_of_words_with_sentiment[
            "bag_of_words_with_sentiment"
        ].apply(pd.Series)
        bag_of_words_with_sentiment = bag_of_words_with_sentiment.rename(
            columns={0: "trade_sentiment"}
        )
        self.bag_of_words_with_sentiment = bag_of_words_with_sentiment
        return


class FinBert(TradePolicyShockSentimentAnalyzer):
    """FinBERT sentiment analyzer inheriting from TradePolicyShockSentimentAnalyzer."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        super().__init__(model_name=model_name)
        print(f"Initialized FinBERT analyzer with model: {model_name}")

    def analyze_transcripts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze trade policy exposure for transcripts using FinBERT.

        Transforms the MultiIndex DataFrame to match the expected format
        for the parent analyzer.
        """
        # Transform DataFrame to match expected format for analyzer
        # Expected: row.iloc[0] = dict with date/ticker, row.iloc[1] = transcript text
        transformed_data = []

        for idx, row in df.iterrows():
            # Extract date and ticker from MultiIndex
            if isinstance(idx, tuple) and len(idx) >= 2:
                date, ticker = idx[0], idx[1]
            else:
                # Fallback if index format is different
                date, ticker = str(idx), "UNKNOWN"

            # Get transcript text
            transcript = row["transcript"] if "transcript" in row else ""

            # Create row in expected format
            transformed_row = [
                {"date": date, "ticker": ticker},  # row.iloc[0]
                transcript,  # row.iloc[1]
            ]
            transformed_data.append(transformed_row)

        # Create DataFrame in expected format
        transformed_df = pd.DataFrame(transformed_data)

        # Call parent analyzer with transformed data
        return self.analyze_dataframe(transformed_df)
