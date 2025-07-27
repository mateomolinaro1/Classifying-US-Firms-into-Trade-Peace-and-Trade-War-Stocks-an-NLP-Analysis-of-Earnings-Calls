# Type checker has issues with pipeline import, but it works at runtime
from transformers import pipeline  # type: ignore
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import re
from tqdm import tqdm
import logging
from dataclasses import dataclass
from .config import TRADE_VOCABULARY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""

    label: str
    score: float
    positive_score: float
    negative_score: float
    neutral_score: float


class TradePolicyShockSentimentAnalyzer:
    """
    Sentiment analyzer for measuring exposure to trade policy shocks
    following the methodology from "Firm-Level Exposure to Trade Policy Shocks:
    A Multi-dimensional Measurement Approach" (Bruno, Goltz and Luyten, 2023)
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer with FinBERT model. Could be
        replaced with any HuggingFace model identifier.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.pipeline = None
        self.trade_keywords = self._load_trade_policy_keywords()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the FinBERT pipeline"""
        try:
            # Type checker has issues but this works at runtime
            self.pipeline = pipeline(  # type: ignore
                "sentiment-analysis",  # type: ignore
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True,
            )
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _load_trade_policy_keywords(self) -> List[str]:
        """
        Load trade policy related keywords for text filtering
        Based on the research methodology for identifying trade-relevant content
        """
        # keywords = [
        #     # Trade policy terms
        #     "tariff",
        #     "tariffs",
        #     "trade war",
        #     "trade policy",
        #     "trade tension",
        #     "trade dispute",
        #     "trade agreement",
        #     "trade deal",
        #     "trade negotiation",
        #     "import",
        #     "imports",
        #     "export",
        #     "exports",
        #     "customs",
        #     "duty",
        #     "duties",
        #     "quota",
        #     "quotas",
        #     # International trade entities
        #     "china",
        #     "chinese",
        #     "wto",
        #     "world trade organization",
        #     "usmca",
        #     "nafta",
        #     "european union",
        #     "eu",
        #     "brexit",
        #     # Supply chain and trade impacts
        #     "supply chain",
        #     "supply chains",
        #     "sourcing",
        #     "procurement",
        #     "manufacturing",
        #     "production",
        #     "logistics",
        #     "shipping",
        #     "raw materials",
        #     "components",
        #     "suppliers",
        #     # Economic impacts
        #     "cost",
        #     "costs",
        #     "pricing",
        #     "price increase",
        #     "margin",
        #     "margins",
        #     "inflation",
        #     "deflation",
        #     "currency",
        #     "exchange rate",
        #     # Policy and regulatory
        #     "regulation",
        #     "regulatory",
        #     "policy",
        #     "government",
        #     "administration",
        #     "sanctions",
        #     "embargo",
        #     "restriction",
        #     "restrictions",
        # ]
        keywords = TRADE_VOCABULARY
        return keywords

    def _extract_trade_relevant_sentences(
        self, text: str, window_size: int = 2
    ) -> List[str]:
        """
        Extract sentences that contain trade-related keywords with context

        Args:
            text: Input text to analyze
            window_size: Number of sentences before/after keyword match to include

        Returns:
            List of trade-relevant sentence segments
        """
        # Handle pandas Series by converting to string
        if isinstance(text, pd.Series):
            if text.empty:
                return []
            text = str(text.iloc[0]) if len(text) > 0 else str(text)
        elif not text or pd.isna(text):
            return []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        trade_segments = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # Check if sentence contains trade keywords
            if any(keyword in sentence_lower for keyword in self.trade_keywords):
                # Extract sentence with context window
                start_idx = max(0, i - window_size)
                end_idx = min(len(sentences), i + window_size + 1)

                segment = " ".join(sentences[start_idx:end_idx])
                trade_segments.append(segment)

        return trade_segments

    def _analyze_segment_sentiment(self, segment: str) -> SentimentResult:
        """
        Analyze sentiment of a text segment using FinBERT

        Args:
            segment: Text segment to analyze

        Returns:
            SentimentResult object with detailed scores
        """
        if not segment or len(segment.strip()) < 10:
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0)

        try:
            # Truncate to model's maximum length (512 tokens for BERT)
            if len(segment) > 2000:  # Conservative estimate for token length
                segment = segment[:2000]

            results = self.pipeline(segment)  # type: ignore

            # Extract scores for each sentiment class
            scores_dict = {
                result["label"].lower(): result["score"] for result in results[0]
            }

            positive_score = scores_dict.get("positive", 0.0)
            negative_score = scores_dict.get("negative", 0.0)
            neutral_score = scores_dict.get("neutral", 0.0)

            # Determine primary sentiment
            max_score = max(positive_score, negative_score, neutral_score)
            if max_score == positive_score:
                primary_label = "positive"
            elif max_score == negative_score:
                primary_label = "negative"
            else:
                primary_label = "neutral"

            return SentimentResult(
                label=primary_label,
                score=max_score,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
            )

        except Exception as e:
            logger.warning(f"Error analyzing segment sentiment: {e}")
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0)

    def calculate_trade_policy_exposure(
        self, date_ticker: Dict[str, Any], transcript: str, min_segments: int = 1
    ) -> Dict[str, Union[str, float, int, None]]:
        """
        Calculate trade policy exposure metrics for a single transcript

        Args:
            date_ticker: Dictionary with date and ticker information
            transcript: Earnings call transcript text
            min_segments: Minimum number of trade-relevant segments required

        Returns:
            Dictionary with exposure metrics
        """
        # Handle pandas Series by converting to string
        if isinstance(transcript, pd.Series):
            if transcript.empty:
                transcript = ""
            else:
                transcript = (
                    str(transcript.iloc[0]) if len(transcript) > 0 else str(transcript)
                )
        elif not transcript:
            transcript = ""

        # Extract trade-relevant segments
        trade_segments = self._extract_trade_relevant_sentences(transcript)

        if len(trade_segments) < min_segments:
            return {
                "date": date_ticker.get("date"),
                "ticker": date_ticker.get("ticker"),
                "trade_exposure_score": 0.0,
                "sentiment_score": 0.0,
                "positive_intensity": 0.0,
                "negative_intensity": 0.0,
                "neutral_intensity": 0.0,
                "num_trade_segments": 0,
                "trade_mention_frequency": 0.0,
            }

        # Analyze sentiment for each segment
        segment_results = []
        for segment in trade_segments:
            result = self._analyze_segment_sentiment(segment)
            segment_results.append(result)

        # Calculate aggregate metrics
        if segment_results:
            avg_positive = np.mean([r.positive_score for r in segment_results])
            avg_negative = np.mean([r.negative_score for r in segment_results])
            avg_neutral = np.mean([r.neutral_score for r in segment_results])

            # Trade policy exposure score (higher negative sentiment = higher exposure)
            trade_exposure_score = float(avg_negative - avg_positive)

            # Overall sentiment score (positive - negative)
            sentiment_score = float(avg_positive - avg_negative)

            # Trade mention frequency (relative to total text length)
            total_trade_text_length = sum(len(segment) for segment in trade_segments)
            total_text_length = len(transcript) if transcript else 1
            trade_mention_frequency = float(total_trade_text_length / total_text_length)

        else:
            avg_positive = avg_negative = avg_neutral = 0.0
            trade_exposure_score = sentiment_score = trade_mention_frequency = 0.0

        return {
            "date": date_ticker.get("date"),
            "ticker": date_ticker.get("ticker"),
            "trade_exposure_score": trade_exposure_score,
            "sentiment_score": sentiment_score,
            "positive_intensity": float(avg_positive),
            "negative_intensity": float(avg_negative),
            "neutral_intensity": float(avg_neutral),
            "num_trade_segments": len(trade_segments),
            "trade_mention_frequency": trade_mention_frequency,
        }

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze trade policy exposure for all transcripts in dataframe

        Args:
            df: DataFrame with columns [0] = date_ticker dict, [1] = transcript text

        Returns:
            DataFrame with exposure metrics for each company-date
        """
        results = []

        logger.info(f"Analyzing {len(df)} transcripts for trade policy exposure...")

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing transcripts"
        ):
            try:
                date_ticker_dict = row.iloc[0] if isinstance(row.iloc[0], dict) else {}
                transcript_text = row.iloc[1] if len(row) > 1 else ""

                # Calculate exposure metrics
                exposure_metrics = self.calculate_trade_policy_exposure(
                    date_ticker=date_ticker_dict, transcript=transcript_text
                )

                results.append(exposure_metrics)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add empty result to maintain dataframe integrity
                results.append(
                    {
                        "date": None,
                        "ticker": None,
                        "trade_exposure_score": 0.0,
                        "sentiment_score": 0.0,
                        "positive_intensity": 0.0,
                        "negative_intensity": 0.0,
                        "neutral_intensity": 0.0,
                        "num_trade_segments": 0,
                        "trade_mention_frequency": 0.0,
                    }
                )

        results_df = pd.DataFrame(results)

        # Log summary statistics
        logger.info("Analysis Summary:")
        logger.info(f"Total transcripts processed: {len(results_df)}")
        logger.info(
            f"Transcripts with trade mentions: {(results_df['num_trade_segments'] > 0).sum()}"
        )
        logger.info(
            f"Average trade exposure score: {results_df['trade_exposure_score'].mean():.4f}"
        )
        logger.info(
            f"Average sentiment score: {results_df['sentiment_score'].mean():.4f}"
        )

        return results_df

    def get_exposure_rankings(
        self, results_df: pd.DataFrame, top_n: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Get rankings of companies by different exposure metrics

        Args:
            results_df: Results dataframe from analyze_dataframe
            top_n: Number of top companies to return for each metric

        Returns:
            Dictionary with different ranking dataframes
        """
        rankings = {}

        # Filter out companies with no trade mentions
        active_companies = results_df[results_df["num_trade_segments"] > 0].copy()

        if len(active_companies) == 0:
            logger.warning("No companies found with trade policy mentions")
            return rankings

        # Highest trade exposure (most negative sentiment about trade)
        rankings["highest_exposure"] = active_companies.nlargest(
            top_n, "trade_exposure_score"
        )[
            [
                "ticker",
                "date",
                "trade_exposure_score",
                "num_trade_segments",
                "trade_mention_frequency",
            ]
        ]

        # Most positive trade sentiment
        rankings["most_positive"] = active_companies.nlargest(top_n, "sentiment_score")[
            [
                "ticker",
                "date",
                "sentiment_score",
                "positive_intensity",
                "num_trade_segments",
            ]
        ]

        # Most negative trade sentiment
        rankings["most_negative"] = active_companies.nsmallest(
            top_n, "sentiment_score"
        )[
            [
                "ticker",
                "date",
                "sentiment_score",
                "negative_intensity",
                "num_trade_segments",
            ]
        ]

        # Highest trade mention frequency
        rankings["highest_frequency"] = active_companies.nlargest(
            top_n, "trade_mention_frequency"
        )[
            [
                "ticker",
                "date",
                "trade_mention_frequency",
                "num_trade_segments",
                "sentiment_score",
            ]
        ]

        return rankings
