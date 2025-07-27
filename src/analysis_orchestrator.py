"""
Main analysis orchestrator for Trade Policy Analysis.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .nlp_models import BagOfWords, BagOfWordsWithSentiment, FinBert
from .descriptive_stats import DescriptiveStatsAnalyzer
from .visualization import TradePolicyVisualizer
from .data_loader import DataLoader, TranscriptTypes
from .config import (
    TRADE_VOCABULARY,
    EVENT_DATES,
    DATA_PATHS,
    OUTPUT_PATHS,
    ANALYSIS_CONFIG,
)

logger = logging.getLogger(__name__)


class TradePolicyAnalyzer:
    """
    Main orchestrator class for trade policy analysis.

    This class coordinates all analysis steps including:
    - Data loading and preprocessing
    - Bag of Words analysis
    - Bag of Words with Sentiment analysis
    - FinBERT analysis
    - Descriptive statistics
    - Visualization generation
    - Event study analysis
    """

    def __init__(
        self,
        output_dir: str = OUTPUT_PATHS["plots"],
        run_finbert: bool = True,
        event_dates: Optional[List[str]] = None,
    ):
        """
        Initialize the Trade Policy Analyzer.

        Args:
            output_dir: Directory for output files
            run_finbert: Whether to run FinBERT analysis
            event_dates: List of event dates for analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_finbert = run_finbert
        self.event_dates = event_dates or EVENT_DATES
        self.trade_vocabulary = TRADE_VOCABULARY

        # Initialize data and results storage
        self.data_loader = DataLoader()
        self.formatted_transcripts_preprocessed: Optional[pd.DataFrame] = None

        # Analysis results
        self.bow_results: Optional[pd.DataFrame] = None
        self.bowws_results: Optional[pd.DataFrame] = None
        self.finbert_results: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}

        # Analysis components
        self.stats_analyzer = DescriptiveStatsAnalyzer()
        self.visualizer = TradePolicyVisualizer(str(self.output_dir))

        logger.info(
            f"TradePolicyAnalyzer initialized with output_dir: {self.output_dir}"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Load preprocessed transcript data using the DataLoader class.

        Returns:
            Loaded transcript DataFrame
        """
        logger.info("Loading transcript data using DataLoader...")

        try:
            # Use the existing DataLoader singleton with proper transcript types
            self.formatted_transcripts_preprocessed = self.data_loader.get_data(
                TranscriptTypes.UNPROCESSED.value
            )

            if self.formatted_transcripts_preprocessed is None:
                raise ValueError("DataLoader returned None")

            logger.info(
                f"Loaded {len(self.formatted_transcripts_preprocessed)} transcripts using DataLoader"
            )
            self._log_data_structure()
            return self.formatted_transcripts_preprocessed

        except Exception as e:
            # Fallback: try to load a subset of chunks for testing if full load fails
            logger.warning(f"Full data loading failed: {e}")
            logger.info("Attempting to load subset of chunks for testing...")

            # Load chunked data as fallback
            chunks_dir = Path(DATA_PATHS["transcripts_chunks"])
            if not chunks_dir.exists():
                raise FileNotFoundError(
                    f"Transcripts directory not found at: {chunks_dir}"
                )

            # Find all chunk files
            chunk_files = list(
                chunks_dir.glob("formatted_transcripts_gzip_chunk_*.pkl")
            )
            if not chunk_files:
                raise FileNotFoundError(f"No chunk files found in: {chunks_dir}")

            logger.info(
                f"Found {len(chunk_files)} chunk files, loading first 5 for testing"
            )

            # Load and combine chunks
            all_chunks = []
            for i, chunk_file in enumerate(
                sorted(chunk_files)[:5]
            ):  # Load first 5 chunks for testing
                logger.info(f"Loading chunk {i+1}/5: {chunk_file.name}")
                try:
                    chunk_data = pd.read_pickle(chunk_file)
                    all_chunks.append(chunk_data)
                except Exception as chunk_e:
                    logger.warning(f"Failed to load chunk {chunk_file.name}: {chunk_e}")

            if not all_chunks:
                raise ValueError("No chunks were successfully loaded")

            # Combine all chunks
            self.formatted_transcripts_preprocessed = pd.concat(
                all_chunks, ignore_index=False
            )

            logger.info(
                f"Loaded {len(self.formatted_transcripts_preprocessed)} transcripts from {len(all_chunks)} chunks (fallback mode)"
            )
            self._log_data_structure()
            return self.formatted_transcripts_preprocessed

    def _log_data_structure(self):
        """Log information about the data structure."""
        data = self.formatted_transcripts_preprocessed
        if data is None:
            logger.warning("No data available to analyze structure")
            return

        logger.info("Analyzing data structure...")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Index type: {type(data.index)}")
        logger.info(f"Index names: {data.index.names}")

        if len(data) > 0:
            sample_idx = data.index[0]
            logger.info(f"Sample index: {sample_idx} (type: {type(sample_idx)})")
            sample_row = data.iloc[0]
            logger.info(f"Sample row columns: {sample_row.index.tolist()}")
            if "transcript" in sample_row:
                logger.info(
                    f"Sample transcript length: {len(str(sample_row['transcript']))}"
                )
        logger.info("---")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete trade policy analysis pipeline.

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete trade policy analysis...")

        try:
            # Step 1: Load data
            self.load_data()

            if self.formatted_transcripts_preprocessed is None:
                raise ValueError("No data loaded for analysis")

            # Step 2: Run BOW analysis
            logger.info("Running Bag of Words analysis...")
            bow = BagOfWords(
                formatted_transcripts_preprocessed=self.formatted_transcripts_preprocessed
            )
            bow.create_bag_of_words_for_transcripts()

            if bow.bag_of_words_frequency_sum is not None:
                self.bow_results = bow.bag_of_words_frequency_sum
                logger.info(
                    f"BOW analysis complete. Results shape: {self.bow_results.shape}"
                )
            else:
                logger.error("BOW analysis failed")

            # Step 3: Run BOWWS analysis
            logger.info("Running Bag of Words with Sentiment analysis...")
            bowws = BagOfWordsWithSentiment(
                formatted_transcripts_preprocessed=self.formatted_transcripts_preprocessed
            )
            bowws.create_bag_of_words_with_sentiment_for_transcripts()

            if bowws.bag_of_words_with_sentiment is not None:
                self.bowws_results = bowws.bag_of_words_with_sentiment
                logger.info(
                    f"BOWWS analysis complete. Results shape: {self.bowws_results.shape}"
                )
            else:
                logger.error("BOWWS analysis failed")

            # Step 4: Run FinBERT analysis if enabled
            if self.run_finbert and self.formatted_transcripts_preprocessed is not None:
                try:
                    logger.info("Running FinBERT Analysis...")

                    # Use the FinBert class with analyze_transcripts method
                    finbert_analyzer = FinBert()
                    self.finbert_results = finbert_analyzer.analyze_transcripts(
                        self.formatted_transcripts_preprocessed
                    )

                    if self.finbert_results is not None:
                        logger.info(
                            f"FinBERT analysis complete. Results shape: {self.finbert_results.shape}"
                        )
                    else:
                        logger.error("FinBERT analysis failed")

                except Exception as e:
                    logger.error(f"FinBERT analysis failed: {str(e)}")
            else:
                logger.info(
                    "FinBERT analysis disabled or no data available, skipping..."
                )

            # Step 5: Generate visualizations if results are available
            if self.bow_results is not None and self.bowws_results is not None:
                logger.info("Generating visualizations...")
                try:
                    # Generate plots using the existing methods
                    self.visualizer.plot_and_save(
                        self.bow_results,
                        "bow_frequency_analysis",
                        "BOW Frequency Analysis",
                    )
                    self.visualizer.plot_and_save(
                        self.bowws_results,
                        "bowws_sentiment_analysis",
                        "BOWWS Sentiment Analysis",
                    )

                    if self.finbert_results is not None:
                        self.visualizer.plot_and_save(
                            self.finbert_results, "finbert_analysis", "FinBERT Analysis"
                        )

                    logger.info("Visualizations generated successfully")
                except Exception as e:
                    logger.warning(f"Error generating visualizations: {str(e)}")

            logger.info("Complete analysis finished successfully")

            # Store results for return
            self.analysis_results = {
                "bow_results": self.bow_results,
                "bowws_results": self.bowws_results,
                "finbert_results": self.finbert_results,
            }

            return self.analysis_results

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all analysis results.

        Returns:
            Summary dictionary
        """
        summary = {
            "data_loaded": self.formatted_transcripts_preprocessed is not None,
            "bow_completed": self.bow_results is not None,
            "bowws_completed": self.bowws_results is not None,
            "finbert_completed": self.finbert_results is not None,
            "output_directory": str(self.output_dir),
        }

        if self.formatted_transcripts_preprocessed is not None:
            summary["total_transcripts"] = len(self.formatted_transcripts_preprocessed)

        if self.bow_results is not None:
            summary["bow_results_shape"] = self.bow_results.shape

        if self.bowws_results is not None:
            summary["bowws_results_shape"] = self.bowws_results.shape

        if self.finbert_results is not None:
            summary["finbert_results_shape"] = self.finbert_results.shape

        return summary
