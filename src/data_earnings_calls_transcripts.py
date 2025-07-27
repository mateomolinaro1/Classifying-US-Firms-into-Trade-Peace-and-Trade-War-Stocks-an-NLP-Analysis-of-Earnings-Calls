import requests
import pandas as pd
from typing import List
from utilities import preprocess_text


class DataEarningsCallsTranscripts:
    """
    Class to handle earnings calls transcripts data.
    """

    def __init__(
        self,
        index_constituents_feather_path: str,
        years: List[int] = None,
        quarters: List[int] = (1, 2, 3, 4),
    ):
        """
        Initializes the DataEarningsCallsTranscripts with the provided data.

        :param tickers: A list of stock tickers for which to retrieve earnings call transcripts.
        :param years: A list of years for which to retrieve earnings call transcripts.
        :param quarters: A list of quarters (1, 2, 3, or 4) for which to retrieve earnings call transcripts.
        """
        self.index_constituents_feather_path = index_constituents_feather_path
        self.years = years
        self.quarters = quarters

        self.index_constituents_feather = None
        self.dates_tickers_api = None
        self.mapping_api_to_ticker = None
        self.transcripts = None
        self.formatted_transcripts = None
        self.formatted_transcripts_preprocessed = None

    def load_index_constituents(self):
        """
        Loads the index constituents from a Feather file.

        :return: A DataFrame containing the index constituents present at each ID_DATE
        """
        self.index_constituents_feather = pd.read_feather(
            r".\data\RIY Index constituents.feather"
        )
        self.index_constituents_feather["Ticker_api"] = (
            self.index_constituents_feather["Ticker"].str.split().str[0]
        )
        self.dates_tickers_api = self.index_constituents_feather[
            ["ID_DATE", "Ticker_api"]
        ].drop_duplicates()

        self.mapping_api_to_ticker = dict(
            zip(
                self.index_constituents_feather["Ticker_api"],
                self.index_constituents_feather["Ticker"],
            )
        )

        return

    @staticmethod
    def get_earnings_transcript(
        ticker: str,
        year: int,
        quarter: int,
        api_key: str = "ZJRxKtR0t6ecUJnBWTuAIg==M6sSdHpeo8Du7fKv",
    ) -> dict:
        """
        Retrieves the earnings call transcript for a specific company and for a specified quarter.

        :param api_key: api key for the API Ninjas Earnings Transcript API.
        :param ticker: The ticker of the company for which to retrieve the transcript.
        :param year: The year of the earnings call.
        :param quarter: The quarter of the earnings call (1, 2, 3, or 4).
        :return: The earnings call transcript for the specified company and the filing date (data at which the transcript
        became available).
        """

        url = "https://api.api-ninjas.com/v1/earningstranscript"

        params = {
            "X-Api-Key": api_key,
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
        }

        response = requests.get(url=url, params=params)
        response.raise_for_status()  # raise error if code different from 200

        response_json = response.json()
        if len(response_json) == 0:
            print(f"No earnings transcript found for {ticker} in {year} Q{quarter}.")
            return {"filing_date": None, "transcript": None}
        else:
            return {
                "filing_date": response_json.get("date"),
                "transcript": response_json.get("transcript"),
            }

    def get_earnings_transcript_for_index_constituents(
        self, years: List[int] = None, quarters: List[int] = (1, 2, 3, 4)
    ):
        """
        Retrieves earnings call transcripts for all index constituents for the specified years and quarters.
        :return: a dict with filling dates as keys, and a dict with tickers as keys and transcripts as values.
        """
        if self.index_constituents_feather is None:
            self.load_index_constituents()

        if years is None:
            years = self.dates_tickers_api["ID_DATE"].dt.year.unique()
        else:
            # Check if provided years are in the index constituents data
            if not set(years).issubset(
                set(self.dates_tickers_api["ID_DATE"].dt.year.unique())
            ):
                raise ValueError(
                    "Provided years are not in the index constituents data."
                )

        if quarters is None:
            quarters = [1, 2, 3, 4]

        # To store
        dct_transcripts = {
            year: {
                quarter: {
                    ticker: {"filing_date": None, "transcript": None}
                    for ticker in self.index_constituents_feather["Ticker_api"].unique()
                }
                for quarter in quarters
            }
            for year in years
        }
        for year in years:
            print(f"Processing year: {year}")
            for quarter in quarters:
                print(f"Processing quarter: {quarter}")
                for ticker_api in self.index_constituents_feather[
                    self.index_constituents_feather["ID_DATE"].dt.year == year
                ]["Ticker_api"]:
                    print(f"Processing ticker: {ticker_api}")
                    transcript = DataEarningsCallsTranscripts.get_earnings_transcript(
                        ticker=ticker_api, year=year, quarter=quarter
                    )
                    dct_transcripts[year][quarter][ticker_api]["filing_date"] = (
                        transcript["filing_date"]
                    )
                    dct_transcripts[year][quarter][ticker_api]["transcript"] = (
                        transcript["transcript"]
                    )

        self.transcripts = dct_transcripts
        self.format_transcripts()
        return

    def format_transcripts(self):
        """
        Formats the transcripts into a MultiIndex DataFrame: (filing_date, ticker_api) as index and 'transcript' as column.
        :return: pd.DataFrame
        """
        years = list(self.transcripts.keys())
        quarters = list(self.transcripts[years[0]].keys())
        tickers_api = self.index_constituents_feather["Ticker_api"].unique()

        data = []

        for year in years:
            for quarter in quarters:
                for ticker_api in tickers_api:
                    entry = (
                        self.transcripts.get(year, {})
                        .get(quarter, {})
                        .get(ticker_api, {})
                    )
                    filing_date = entry.get("filing_date")
                    transcript = entry.get("transcript")

                    if filing_date is not None:
                        data.append((filing_date, ticker_api, transcript))

        # Creation of the MultiIndex df
        df = pd.DataFrame(data, columns=["filing_date", "ticker_api", "transcript"])
        df = df.set_index(["filing_date", "ticker_api"]).sort_index()

        self.formatted_transcripts = df
        return

    def preprocess_transcripts(
        self,
        column_name_to_clean: str = "transcript",
        new_column_name: str = "transcript_cleaned",
    ):
        """
        Preprocesses the transcripts in the DataFrame by tokenizing, removing stopwords, and stemming.

        :param column_name_to_clean: The name of the column containing the transcripts to clean.
        :param new_column_name: The name of the new column to store the cleaned transcripts.
        :return: DataFrame with an additional 'transcript_cleaned' column.
        """
        if self.formatted_transcripts is None:
            raise ValueError(
                "Transcripts have not been formatted yet. Please call format_transcripts() first."
            )

        self.formatted_transcripts_preprocessed = preprocess_text(
            self.formatted_transcripts,
            column_name_to_clean=column_name_to_clean,
            new_column_name=new_column_name,
        )
        return
