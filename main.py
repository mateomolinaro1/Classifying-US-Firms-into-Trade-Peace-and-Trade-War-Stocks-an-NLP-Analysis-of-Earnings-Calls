import logging
from pathlib import Path

from src.analysis_orchestrator import TradePolicyAnalyzer
from src.config import OUTPUT_PATHS, EVENT_DATES


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(OUTPUT_PATHS["plots"]) / "analysis.log"),
        ],
    )


def main():
    """
    Main analysis function using the TradePolicyAnalyzer orchestrator.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STARTING TRADE POLICY ANALYSIS PIPELINE")
    logger.info("=" * 60)

    try:
        # Initialize the analyzer with configuration
        analyzer = TradePolicyAnalyzer(
            output_dir=OUTPUT_PATHS["plots"],
            run_finbert=True,  # Enable FinBERT analysis
            event_dates=EVENT_DATES,
        )

        # Run the complete analysis pipeline
        results = analyzer.run_complete_analysis()

        # Print summary of results
        summary = analyzer.get_results_summary()
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE - SUMMARY:")
        logger.info("=" * 60)

        for key, value in summary.items():
            logger.info(f"{key}: {value}")

        # Optional: Display specific results
        if "events" in results:
            logger.info("\nEvent Analysis Results:")
            for event_type, event_data in results["events"].items():
                if hasattr(event_data, "shape"):
                    logger.info(f"  {event_type}: {event_data.shape}")
                else:
                    logger.info(
                        f"  {event_type}: {len(event_data) if event_data else 0} items"
                    )

        logger.info(f"\nOutputs saved to: {analyzer.output_dir}")
        logger.info("Analysis pipeline completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}")
        logger.exception("Full error traceback:")
        raise


if __name__ == "__main__":
    main()
