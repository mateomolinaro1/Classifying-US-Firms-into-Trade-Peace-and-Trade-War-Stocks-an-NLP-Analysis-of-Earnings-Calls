import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DescriptiveStatsAnalyzer:
    """
    A comprehensive class for computing descriptive statistics on trade policy data.
    """

    def __init__(self):
        """Initialize the DescriptiveStatsAnalyzer."""
        pass

    @staticmethod
    def compute_mean_and_monthly_mean_freq(
        bag_of_words_frequency_sum: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute daily and monthly mean frequencies from bag of words data.

        Args:
            bag_of_words_frequency_sum: DataFrame with bag of words frequency data

        Returns:
            Dictionary containing daily and monthly mean frequency DataFrames
        """
        df_mean_daily = bag_of_words_frequency_sum.groupby(level="filing_date").mean()
        df_mean_daily.index = pd.to_datetime(df_mean_daily.index)
        df_mean_monthly = df_mean_daily.resample("ME").mean()

        logger.info(f"Computed daily means for {len(df_mean_daily)} days")
        logger.info(f"Computed monthly means for {len(df_mean_monthly)} months")

        return {"df_mean_daily": df_mean_daily, "df_mean_monthly": df_mean_monthly}

    @staticmethod
    def count_companies_talking_at_least_once_about_tariffs_per_quarter(
        bag_of_words_frequency_sum: pd.DataFrame,
        condition_column: str,
        bow_or_bowws: str = "bow",
    ) -> Dict[str, pd.DataFrame]:
        """
        Count companies mentioning trade topics at least once per quarter.

        Args:
            bag_of_words_frequency_sum: DataFrame with trade frequency data
            condition_column: Column name to use for the condition
            bow_or_bowws: Type of analysis ("bow" or "bowws")

        Returns:
            Dictionary containing count results and quarterly data
        """
        bow_freq = bag_of_words_frequency_sum.copy()
        bow_freq.reset_index(inplace=True)
        bow_freq["filing_date"] = pd.to_datetime(bow_freq["filing_date"])
        bow_freq["quarter"] = bow_freq["filing_date"].dt.to_period("Q")
        bow_q = bow_freq.groupby(["quarter", "ticker_api"]).last()
        bow_q.reset_index(inplace=True)

        if bow_or_bowws == "bow":
            bow_q["signal_once"] = (bow_q[condition_column] > 0.0).astype(float)
            count_once = bow_q.groupby("quarter")[["signal_once"]].sum()
        elif bow_or_bowws == "bowws":
            bow_q["signal_positive"] = (bow_q[condition_column] > 0.0).astype(float)
            bow_q["signal_negative"] = (bow_q[condition_column] < 0.0).astype(float)
            count_once = pd.DataFrame()
            count_once["positive"] = bow_q.groupby("quarter")[["signal_positive"]].sum()
            count_once["negative"] = bow_q.groupby("quarter")[["signal_negative"]].sum()
        elif bow_or_bowws == "custom_model":
            bow_q["positive"] = (bow_q[condition_column] == "positive").astype(float)
            bow_q["negative"] = (bow_q[condition_column] == "negative").astype(float)
            bow_q["neutral"] = (bow_q[condition_column] == "neutral").astype(float)
            count_once = pd.DataFrame()
            count_once["positive"] = bow_q.groupby("quarter")[["positive"]].sum()
            count_once["negative"] = bow_q.groupby("quarter")[["negative"]].sum()
            count_once["neutral"] = bow_q.groupby("quarter")[["neutral"]].sum()
        else:
            raise ValueError("bow_or_bowws must be either 'bow','bowws' or 'custom_model.")

        # Convert period index to timestamp
        if hasattr(count_once.index, "to_timestamp"):
            count_once.index = count_once.index.to_timestamp()

        logger.info(
            f"Counted companies for {len(count_once)} quarters using {bow_or_bowws} method"
        )

        return {"count_once": count_once, "bow_q": bow_q}

    @staticmethod
    def count_on_events(
        count_once: pd.DataFrame, event_dates: Tuple[str, ...]
    ) -> pd.DataFrame:
        """
        Count trade mentions during specific event periods.

        Args:
            count_once: DataFrame with quarterly counts
            event_dates: Tuple of event date strings

        Returns:
            DataFrame with counts filtered to event periods
        """
        dates_dt = pd.to_datetime(event_dates)
        quarters = dates_dt.to_period("Q")
        prev_quarters = quarters - 1
        prev_quarters_start = prev_quarters.asfreq("Q").start_time
        count_once_at_events = count_once.loc[
            count_once.index.isin(prev_quarters_start)
        ]

        logger.info(
            f"Found {len(count_once_at_events)} quarters matching event periods"
        )

        return count_once_at_events

    def generate_summary_statistics(
        self,
        bow_results: pd.DataFrame,
        bowws_results: pd.DataFrame,
        finbert_results: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for all analysis methods.

        Args:
            bow_results: Bag of Words results
            bowws_results: Bag of Words with Sentiment results
            finbert_results: FinBERT results (optional)

        Returns:
            Dictionary with summary statistics
        """
        summary = {"bow": {}, "bowws": {}, "finbert": {}}

        # BOW statistics
        if bow_results is not None and not bow_results.empty:
            summary["bow"] = {
                "total_transcripts": len(bow_results),
                "avg_frequency": (
                    bow_results["sum"].mean() if "sum" in bow_results.columns else 0
                ),
                "transcripts_with_mentions": (
                    (bow_results["sum"] > 0).sum()
                    if "sum" in bow_results.columns
                    else 0
                ),
            }

        # BOWWS statistics
        if bowws_results is not None and not bowws_results.empty:
            sentiment_col = "trade_sentiment"
            if sentiment_col in bowws_results.columns:
                summary["bowws"] = {
                    "total_transcripts": len(bowws_results),
                    "avg_sentiment": bowws_results[sentiment_col].mean(),
                    "positive_sentiment_count": (
                        bowws_results[sentiment_col] > 0
                    ).sum(),
                    "negative_sentiment_count": (
                        bowws_results[sentiment_col] < 0
                    ).sum(),
                    "neutral_sentiment_count": (
                        bowws_results[sentiment_col] == 0
                    ).sum(),
                }

        # FinBERT statistics
        if finbert_results is not None and not finbert_results.empty:
            transcripts_with_trade = (finbert_results["num_trade_segments"] > 0).sum()
            summary["finbert"] = {
                "total_transcripts": len(finbert_results),
                "transcripts_with_trade_mentions": transcripts_with_trade,
                "avg_trade_exposure_score": finbert_results[
                    "trade_exposure_score"
                ].mean(),
                "avg_sentiment_score": finbert_results["sentiment_score"].mean(),
                "avg_positive_intensity": finbert_results["positive_intensity"].mean(),
                "avg_negative_intensity": finbert_results["negative_intensity"].mean(),
            }

            if transcripts_with_trade > 0:
                exposure_scores = finbert_results[
                    finbert_results["num_trade_segments"] > 0
                ]["trade_exposure_score"]
                summary["finbert"]["exposure_score_range"] = {
                    "min": exposure_scores.min(),
                    "max": exposure_scores.max(),
                    "std": exposure_scores.std(),
                }

        return summary


# Backward compatibility - keep the original function names
def compute_mean_and_monthly_mean_freq(bag_of_words_frequency_sum: pd.DataFrame):
    """Legacy function for backward compatibility."""
    analyzer = DescriptiveStatsAnalyzer()
    return analyzer.compute_mean_and_monthly_mean_freq(bag_of_words_frequency_sum)


def count_companies_talking_at_least_once_about_tariffs_per_quarter(
    bag_of_words_frequency_sum: pd.DataFrame,
    condition_column: str,
    bow_or_bowws: str = "bow",
):
    """Legacy function for backward compatibility."""
    analyzer = DescriptiveStatsAnalyzer()
    return analyzer.count_companies_talking_at_least_once_about_tariffs_per_quarter(
        bag_of_words_frequency_sum, condition_column, bow_or_bowws
    )


def count_on_events(count_once: pd.DataFrame, event_dates: Tuple[str, ...]):
    """Legacy function for backward compatibility."""
    analyzer = DescriptiveStatsAnalyzer()
    return analyzer.count_on_events(count_once, event_dates)
