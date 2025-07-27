import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class TradePolicyVisualizer:
    """
    A comprehensive class for creating visualizations of trade policy analysis results.
    """

    def __init__(self, output_dir: str = "outputs/descriptive_statistics_plots"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Base directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_and_save(
        self,
        df_to_plot: pd.DataFrame,
        title: str,
        save_path: Optional[str] = None,
        font_size: int = 10,
        x_label: str = "Date",
        y_label: str = "Mean Frequency",
        legend: bool = False,
        figsize: tuple = (12, 6),
        dpi: int = 300,
    ) -> str:
        """
        Create and save a plot.

        Args:
            df_to_plot: DataFrame to plot
            title: Plot title
            save_path: Custom save path (optional)
            font_size: Title font size
            x_label: X-axis label
            y_label: Y-axis label
            legend: Whether to show legend
            figsize: Figure size tuple
            dpi: Plot resolution

        Returns:
            Path where the plot was saved
        """
        if save_path is None:
            # Generate default filename from title
            filename = (
                title.lower()
                .replace(" ", "_")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
                + ".png"
            )
            save_path = str(self.output_dir / filename)

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=figsize)
        plt.plot(df_to_plot)
        plt.title(title, fontsize=font_size)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if legend:
            plt.legend(df_to_plot.columns.tolist())

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot to {save_path}")
        return save_path

    def create_bow_visualizations(
        self,
        df_mean_daily: pd.DataFrame,
        df_mean_monthly: pd.DataFrame,
        count_once: pd.DataFrame,
    ) -> List[str]:
        """
        Create all Bag of Words visualizations.

        Args:
            df_mean_daily: Daily frequency data
            df_mean_monthly: Monthly frequency data
            count_once: Quarterly count data

        Returns:
            List of saved plot paths
        """
        saved_paths = []

        # Daily frequency plot
        path1 = self.plot_and_save(
            df_to_plot=df_mean_daily,
            title="Mean Daily Frequency of Trade Policy Words in Russell 1000 Earnings Calls",
            save_path=str(
                self.output_dir / "mean_daily_frequency_trade_policy_words_bow.png"
            ),
        )
        saved_paths.append(path1)

        # Monthly frequency plot
        path2 = self.plot_and_save(
            df_to_plot=df_mean_monthly,
            title="Mean Monthly Frequency of Trade Policy Words in Russell 1000 Earnings Calls",
            save_path=str(
                self.output_dir / "mean_monthly_frequency_trade_policy_words_bow.png"
            ),
        )
        saved_paths.append(path2)

        # Company count plot
        path3 = self.plot_and_save(
            df_to_plot=count_once,
            title="Count of Enterprises mentioning at least one word related to tariffs",
            save_path=str(self.output_dir / "count_trade_policy_words_bow.png"),
            y_label="Count of Enterprises",
            x_label="Quarter",
            legend=True,
        )
        saved_paths.append(path3)

        return saved_paths

    def create_bowws_visualizations(
        self,
        df_mean_daily: pd.DataFrame,
        df_mean_monthly: pd.DataFrame,
        count_once: pd.DataFrame,
    ) -> List[str]:
        """
        Create all Bag of Words with Sentiment visualizations.

        Args:
            df_mean_daily: Daily sentiment data
            df_mean_monthly: Monthly sentiment data
            count_once: Quarterly sentiment count data

        Returns:
            List of saved plot paths
        """
        saved_paths = []

        # Daily sentiment plot
        path1 = self.plot_and_save(
            df_to_plot=df_mean_daily,
            title="Mean Daily Trade Sentiment of Trade Policy Words",
            save_path=str(
                self.output_dir / "mean_daily_frequency_trade_policy_words_bowws.png"
            ),
            y_label="Trade Sentiment",
        )
        saved_paths.append(path1)

        # Monthly sentiment plot
        path2 = self.plot_and_save(
            df_to_plot=df_mean_monthly,
            title="Mean Monthly Trade Sentiment of Trade Policy Words",
            save_path=str(
                self.output_dir / "mean_monthly_frequency_trade_policy_words_bowws.png"
            ),
            y_label="Trade Sentiment",
        )
        saved_paths.append(path2)

        # Sentiment count plot
        path3 = self.plot_and_save(
            df_to_plot=count_once,
            title="Count of Companies by Trade Sentiment (Positive/Negative)",
            save_path=str(self.output_dir / "count_trade_sentiment_bowws.png"),
            y_label="Count of Companies",
            x_label="Quarter",
            legend=True,
        )
        saved_paths.append(path3)

        return saved_paths

    def create_finbert_visualizations(self, finbert_results: pd.DataFrame) -> List[str]:
        """
        Create FinBERT visualizations.

        Args:
            finbert_results: FinBERT analysis results

        Returns:
            List of saved plot paths
        """
        saved_paths = []

        if finbert_results.empty:
            logger.warning("No FinBERT results to visualize")
            return saved_paths

        # Check if there are trade mentions
        transcripts_with_trade = (finbert_results["num_trade_segments"] > 0).sum()
        if transcripts_with_trade == 0:
            logger.warning("No trade policy segments found in FinBERT results")
            return saved_paths

        # Prepare data
        finbert_results_with_date = finbert_results.copy()
        finbert_results_with_date["date"] = pd.to_datetime(
            finbert_results_with_date["date"]
        )
        finbert_results_with_date = finbert_results_with_date.dropna(subset=["date"])

        if finbert_results_with_date.empty:
            logger.warning("No valid dates in FinBERT results")
            return saved_paths

        # Trade exposure plot
        finbert_daily_exposure = finbert_results_with_date.groupby("date")[
            "trade_exposure_score"
        ].mean()
        finbert_monthly_exposure = (
            finbert_daily_exposure.resample("ME").mean().to_frame()
        )

        path1 = self.plot_and_save(
            df_to_plot=finbert_monthly_exposure,
            title="Mean Monthly Trade Policy Exposure Score (FinBERT)",
            save_path=str(self.output_dir / "mean_monthly_trade_exposure_finbert.png"),
            y_label="Trade Exposure Score",
        )
        saved_paths.append(path1)

        # Sentiment distribution plot
        finbert_daily_sentiment = finbert_results_with_date.groupby("date")[
            "sentiment_score"
        ].mean()
        finbert_monthly_sentiment = (
            finbert_daily_sentiment.resample("ME").mean().to_frame()
        )

        path2 = self.plot_and_save(
            df_to_plot=finbert_monthly_sentiment,
            title="Mean Monthly Trade Policy Sentiment Score (FinBERT)",
            save_path=str(self.output_dir / "mean_monthly_trade_sentiment_finbert.png"),
            y_label="Sentiment Score",
        )
        saved_paths.append(path2)

        return saved_paths


# Backward compatibility function
def plot_and_save(
    df_to_plot: pd.DataFrame,
    title: str,
    font_size: int = 10,
    x_label: str = "Date",
    y_label: str = "Mean Frequency",
    legend: bool = False,
    save_path: str = r".\outputs\descriptive_statistics_plots\mean_daily_frequency_trade_policy_words.png",
):
    """Legacy function for backward compatibility."""
    visualizer = TradePolicyVisualizer()
    return visualizer.plot_and_save(
        df_to_plot=df_to_plot,
        title=title,
        save_path=save_path,
        font_size=font_size,
        x_label=x_label,
        y_label=y_label,
        legend=legend,
    )
