"""
Main analysis orchestrator for Trade Policy Analysis.
"""

import pandas as pd
import numpy as np
import logging
import gzip
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from .nlp_models import BagOfWords, BagOfWordsWithSentiment, FinBert, CustomModels
from .data_earnings_calls_transcripts import DataEarningsCallsTranscripts
from .descriptive_stats import DescriptiveStatsAnalyzer
from .visualization import TradePolicyVisualizer
from .data_loader import DataLoader, TranscriptTypes
from sklearn.linear_model import LogisticRegression
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
        run_custom_models: bool = True,
        event_dates: Optional[List[str]] = None,
    ):
        """
        Initialize the Trade Policy Analyzer.

        Args:
            output_dir: Directory for output files
            run_finbert: Whether to run FinBERT analysis
            run_custom_models: Whether to run custom models analysis
            event_dates: List of event dates for analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_finbert = run_finbert
        self.run_custom_models = run_custom_models
        self.event_dates = event_dates or EVENT_DATES
        self.trade_vocabulary = TRADE_VOCABULARY

        # Initialize data and results storage
        self.data_loader = DataLoader()
        self.formatted_transcripts_preprocessed: Optional[pd.DataFrame] = None
        self.formatted_transcripts: Optional[pd.DataFrame] = None


        # Analysis results
        self.bow_results: Optional[pd.DataFrame] = None
        self.bowws_results: Optional[pd.DataFrame] = None
        self.finbert_results: Optional[pd.DataFrame] = None
        self.custom_models_results: Optional[Dict[str, Any]] = None
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

    def run_custom_models_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run custom models analysis including zero-shot classification and logistic regression.

        Returns:
            Dictionary containing custom models results
        """
        logger.info("Starting Custom Models Analysis...")

        try:
            if self.formatted_transcripts_preprocessed is None:
                logger.error("No transcript data available for custom models analysis")
                return None

            # Initialize CustomModels with date ranges
            cm = CustomModels(
                formatted_transcripts=self.formatted_transcripts_preprocessed,
                start_date_training="2007-05-10",
                end_date_training="2019-01-01",
                start_date_validation="2019-01-02",
                end_date_validation="2022-01-01",
                start_date_test="2022-01-02",
                end_date_test="2024-07-24",
            )

            # Step 1: Extract sentences of interest
            logger.info("Extracting sentences of interest...")
            cm.apply_retrieve_sentences_fast()
            cm.get_unlabelled_data_flat()

            # Step 2: Load zero-shot classification results
            logger.info("Loading zero-shot classification results...")
            zero_shot_file_paths = [
                os.path.join(
                    "new_commit_colleague",
                    "data",
                    "zero_shot_classification_sentence_level_labels_train_val.pkl.gz",
                ),
                os.path.join(
                    "data",
                    "zero_shot_classification_sentence_level_labels_train_val.pkl.gz",
                ),
            ]

            loaded_zero_shot = False
            for zero_shot_path in zero_shot_file_paths:
                if os.path.exists(zero_shot_path):
                    logger.info(
                        f"Loading zero-shot classification results from {zero_shot_path}..."
                    )
                    with gzip.open(zero_shot_path, "rb") as handle:
                        cm.sentence_level_labels = pickle.load(handle)
                    loaded_zero_shot = True
                    break

            if not loaded_zero_shot:
                logger.warning(
                    "Zero-shot classification results not found. Skipping custom models analysis."
                )
                logger.info(
                    "You may need to run zero_shot_classification_sentence_level() first (takes ~25 minutes)."
                )
                return None

            # Step 3: Get accuracy and optimal threshold
            logger.info("Computing zero-shot classification accuracy...")
            cm.get_accuracy_zero_shot_classification(
                human_label_df=None,
                loading_path_human=os.path.join(
                    "outputs", "zero_shot_classification_results_human_label.xlsx"
                ),
                file_extension="xlsx",
                usecols="A:H",
                threshold_range=(0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            )

            # Step 4: Filter sentences by threshold
            logger.info("Filtering sentences by optimal threshold...")
            cm.get_zsc_sentence_level_label_filtered_threshold(
                threshold=cm.optimal_threshold_zsc
            )

            # Log data range information
            if cm.sentence_level_label_filtered_threshold is not None:
                date_min = (
                    cm.sentence_level_label_filtered_threshold.index.get_level_values(
                        "filing_date"
                    ).min()
                )
                date_max = (
                    cm.sentence_level_label_filtered_threshold.index.get_level_values(
                        "filing_date"
                    ).max()
                )
                logger.info(f"Date range in filtered data: {date_min} to {date_max}")
                logger.info(
                    f"Training period: {cm.start_date_training} to {cm.end_date_training}"
                )
                logger.info(f"Test period: {cm.start_date_test} to {cm.end_date_test}")

            # Step 5: Split data and train model
            logger.info("Splitting data and training logistic regression model...")
            cm.split_train_validation_test()
            cm.fit_model(
                LogisticRegression,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
            )

            # Collect results
            results = {
                "accuracy_metrics": cm.accuracy_zero_shot_classification,
                "optimal_threshold": cm.optimal_threshold_zsc,
                "filtered_data_shape": (
                    cm.sentence_level_label_filtered_threshold.shape
                    if cm.sentence_level_label_filtered_threshold is not None
                    else None
                ),
                "model_trained": cm.clf is not None,
                "sentences_extracted": cm.sentences_of_interest_df is not None,
            }

            logger.info("Custom Models Analysis completed successfully")
            logger.info(f"Optimal threshold: {cm.optimal_threshold_zsc}")
            if cm.sentence_level_label_filtered_threshold is not None:
                logger.info(
                    f"Filtered data shape: {cm.sentence_level_label_filtered_threshold.shape}"
                )

            self.custom_models_results = results
            return results

        except Exception as e:
            logger.error(f"Custom Models Analysis failed: {str(e)}")
            logger.exception("Full error traceback:")
            return None

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

            # Step 5: Run Custom Models analysis if enabled
            if (
                self.run_custom_models
                and self.formatted_transcripts_preprocessed is not None
            ):
                logger.info("Running Custom Models Analysis...")
                self.custom_models_results = self.run_custom_models_analysis()
            else:
                logger.info(
                    "Custom Models analysis disabled or no data available, skipping..."
                )

            # Step 6: Generate visualizations if results are available
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
                "custom_models_results": self.custom_models_results,
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
            "custom_models_completed": self.custom_models_results is not None,
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

        if self.custom_models_results is not None:
            summary["custom_models_optimal_threshold"] = self.custom_models_results.get(
                "optimal_threshold"
            )
            summary["custom_models_filtered_shape"] = self.custom_models_results.get(
                "filtered_data_shape"
            )

        return summary
