import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
from typing import List, Tuple
import descriptive_stats as ds

class AbnormalReturnsAnalysisOnEvents:
    def __init__(self,
                 asset_returns_path:str=r'.\data\RIY Index returns.feather',
                 market_returns_path:str=r'.\data\total_returns_russell.feather',
                 rf_returns_path:str=r'.\data\rf_returns.csv',
                 universe_path:str=r'.\data\RIY Index constituents.feather',
                 events_dates:Tuple[str]= ("2017-04-24",
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
                                          "2025-07-23"),
                 nb_periods_before_event_as_start_estimation_normal_returns:int=252,
                 nb_periods_before_event_as_end_estimation_normal_returns:int=7,
                 event_window:int=7
                 ):
        """
        Initializes the AbnormalReturnsAnalysisOnEvents with the provided data.
        :param asset_returns_path:
        :param market_returns_path:
        :param rf_returns_path:
        :param events_dates:
        :param nb_periods_before_event_as_start_estimation_normal_returns:
        :param nb_periods_before_event_as_end_estimation_normal_returns:
        """
        self.asset_returns_path = asset_returns_path
        self.market_returns_path = market_returns_path
        self.rf_returns_path = rf_returns_path
        self.universe_path = universe_path
        self.events_dates = events_dates
        self.nb_periods_before_event_as_start_estimation_normal_returns = nb_periods_before_event_as_start_estimation_normal_returns
        self.nb_periods_before_event_as_end_estimation_normal_returns = nb_periods_before_event_as_end_estimation_normal_returns
        self.event_window = event_window

        self.asset_returns = None
        self.market_returns = None
        self.rf_returns = None
        self.universe = None

        self.event_windows = None
        self.estimation_windows = None
        self.estimation_ranges = None

        self.universe_dct = None
        self.betas = None

        self.normal_returns = None
        self.abnormal_returns = None
        self.cumulative_abnormal_returns_by_event = None
        self.cumulative_abnormal_returns_across_events = None

        # self.mean_cumulative_abnormal_returns_trade_peace_war_by_event = None

    def load_data(self):
        """Loads asset, market, and risk-free returns data from feather/csv files."""
        for path in [self.asset_returns_path, self.market_returns_path, self.rf_returns_path, self.universe_path]:
            match = re.search(r'\.([^.\\/:*?"<>|\r\n]+)$', path)
            if match:
                word_after_dot = match.group(1)
            else:
                raise ValueError("No extension found in the path:", path)

            if word_after_dot == "feather":
                if path == self.asset_returns_path:
                    self.asset_returns = pd.read_feather(path)
                elif path == self.market_returns_path:
                    self.market_returns = pd.read_feather(path)
                elif path == self.rf_returns_path:
                    self.rf_returns = pd.read_feather(path)
                elif path == self.universe_path:
                    self.universe = pd.read_feather(path)
            elif word_after_dot == "csv":
                if path == self.asset_returns_path:
                    self.asset_returns = pd.read_csv(path)
                elif path == self.market_returns_path:
                    self.market_returns = pd.read_csv(path)
                elif path == self.rf_returns_path:
                    self.rf_returns = pd.read_csv(path)
                elif path == self.universe_path:
                    self.universe = pd.read_feather(path)
            elif word_after_dot == "xlsx":
                if path == self.asset_returns_path:
                    self.asset_returns = pd.read_excel(path)
                elif path == self.market_returns_path:
                    self.market_returns = pd.read_excel(path)
                elif path == self.rf_returns_path:
                    self.rf_returns = pd.read_excel(path)
                elif path == self.universe_path:
                    self.universe = pd.read_feather(path)
            else:
                raise ValueError(f"Unsupported file format: {word_after_dot}. Supported formats are feather, csv, and xlsx.")

        return

    def align_data(self,
                   join_type:str='inner'):
        """Aligns the asset, market, and risk-free returns data on the same dates."""
        # self.asset_returns = self.asset_returns.set_index('date')
        # self.market_returns = self.market_returns.set_index('date')
        # self.rf_returns = self.rf_returns.set_index('date')
        self.asset_returns = self.asset_returns[~self.asset_returns.index.duplicated(keep='first')]
        self.market_returns = self.market_returns[~self.market_returns.index.duplicated(keep='first')]
        self.rf_returns = self.rf_returns[~self.rf_returns.index.duplicated(keep='first')]

        # Step 1: Compute combined index (outer or inner)
        if join_type == 'outer':
            combined_index = self.asset_returns.index.union(self.market_returns.index).union(self.rf_returns.index)
        elif join_type == 'inner':
            combined_index = self.asset_returns.index.intersection(self.market_returns.index).intersection(self.rf_returns.index)
        else:
            raise ValueError("join_type must be either 'outer' or 'inner'.")

        # Step 2: Reindex each DataFrame to the combined index
        self.asset_returns = self.asset_returns.reindex(combined_index)
        self.market_returns = self.market_returns.reindex(combined_index)
        self.rf_returns = self.rf_returns.reindex(combined_index)
        return

    # market_returns = pd.read_feather(r'.\data\total_returns_russell.feather')
    # asset_returns = pd.read_feather(r'.\data\RIY Index returns.feather')
    # combined_index = asset_returns.index.intersection(market_returns.index)
    # market_returns = market_returns.reindex(combined_index)
    # asset_returns = asset_returns.reindex(combined_index)
    # rf_returns = pd.DataFrame(data=0.02/360, index=combined_index, columns=['rf_returns'])
    # universe = pd.read_feather(r'.\data\RIY Index constituents.feather')

    # events_dates = ["2017-04-24",
    #                 "2017-08-08",
    #                 "2018-01-01",
    #                 "2018-03-01",
    #                 "2018-03-22",
    #                 "2018-04-02",
    #                 "2018-06-15",
    #                 "2018-09-17",
    #                 "2019-05-10",
    #                 "2019-08-23",
    #                 "2019-09-01",
    #                 "2019-12-15",
    #                 "2020-01-15",
    #                 "2024-05-14",
    #                 "2025-02-01",
    #                 "2025-02-04",
    #                 "2025-03-04",
    #                 "2025-03-12",
    #                 "2025-07-23"]

    def compute_date_ranges_for_regression(self):
        """
        Computes the date ranges used for regressions to estimate the betas for computing the normal returns.
        The normal return is calculated as the rf + beta*market returns as in "Firm-Level Exposure to Trade Policy Shocks:
        A Multi-dimensional Measurement Approach" (June 2023) by Bruno, Goltz and Luyten.
        """
        # Step 1: estimate the betas for each stocks during normal periods.
        # A normal period is defined as 252 days before the date of an event until 7 days before the event.
        # If during these normal periods, other events occur, they're removed for the estimation of the betas.
        # We'll be left with number of stocks * number of events betas.

        # Step 1a: for each event date compute the event window which corresponds to a period that includes the event day
        # plus the event_window periods after the event.
        event_windows = {event_date: [] for event_date in self.events_dates}
        for event_date in self.events_dates:
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            start_event_window = event_date_datetime
            end_event_window = event_date_datetime + pd.Timedelta(days=self.event_window + 2)
            mask = (self.asset_returns.index >= start_event_window) & (self.asset_returns.index <= end_event_window)
            event_windows[event_date] = self.asset_returns[mask].index.tolist()
        self.event_windows = event_windows

        # Step 1b: for each event date, retrieve the start and end date of estimation period.
        estimation_windows = {event_date: (None, None) for event_date in self.events_dates}
        for event_date in self.events_dates:
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            start_estimation = event_date_datetime - pd.Timedelta(days=self.nb_periods_before_event_as_start_estimation_normal_returns)
            end_estimation = event_date_datetime - pd.Timedelta(days=self.nb_periods_before_event_as_end_estimation_normal_returns)
            estimation_windows[event_date] = (start_estimation, end_estimation)
        self.estimation_windows = estimation_windows

        # Step 1c: for each event date, retrieve the date range and remove events that occur during the estimation period.
        estimation_ranges = {event_date: [] for event_date in self.events_dates}
        for event_date, (start_estimation, end_estimation) in self.estimation_windows.items():
            if start_estimation is not None and end_estimation is not None:
                mask = (self.asset_returns.index >= start_estimation) & (self.asset_returns.index <= end_estimation)
                estimation_ranges[event_date] = self.asset_returns[mask].index.tolist()
                # Remove dates that are in the event windows
                estimation_ranges[event_date] = [
                    date
                    for date in estimation_ranges[event_date]
                    if all(date not in event_windows[event_datee] for event_datee in self.events_dates)
                ]
        self.estimation_ranges = estimation_ranges

        return

    def compute_universe(self):
        # We'll store the betas only for the stocks present in the universe at each of the event date.
        # To do that, we need to create a dict with all the (asset) dates as keys and the tickers in a list as values.
        universe_dct = {date: [] for date in self.asset_returns.index}
        for key in universe_dct.keys():
            target_date = key
            dates_universe = self.universe["ID_DATE"]
            closest_inferior_date = dates_universe[dates_universe <= target_date].max()
            tickers = self.universe[self.universe["ID_DATE"] == closest_inferior_date]["Ticker"].tolist()
            universe_dct[key] = tickers  # assign directly
        self.universe_dct = universe_dct

    def regress(self):
        """
        Performs regressions for each stock and for each event date to estimate the betas which will be used to compute
        the normal returns as in "Firm-Level Exposure to Trade Policy Shocks:
        A Multi-dimensional Measurement Approach" (June 2023) by Bruno, Goltz and Luyten.
        :return: stores the betas in self.betas as a dict with event dates as keys and a DataFrame (rows are betas,
        columns are stocks) with the betas
        """
        # To store the betas, we will create a dict with event dates as keys and a DataFrame with betas as values.
        betas = {}
        available_dates = sorted(self.universe_dct.keys())
        for event_date in self.events_dates:
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            closest_date = max([d for d in available_dates if d <= event_date_datetime], default=None)

            if closest_date is not None:
                tickers = self.universe_dct[closest_date]
            else:
                tickers = []

            betas[event_date] = pd.DataFrame(data=np.nan, index=["beta"], columns=tickers)

        # Perform linear regression for each event_date and for each stock in the universe.
        for event_date, estimation_range in self.estimation_ranges.items():
            print(f"Processing event date: {event_date}")
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            closest_date = max([d for d in available_dates if d <= event_date_datetime], default=None)
            if len(estimation_range) < 2:
                continue

            for ticker in self.universe_dct[closest_date]:
                if ticker not in self.asset_returns.columns:
                    continue
                # Prepare the data for regression
                asset_returns_reg = self.asset_returns[ticker].loc[estimation_range]
                market_returns_reg = self.market_returns[self.market_returns.columns[0]].loc[estimation_range]
                rf_returns_reg = self.rf_returns[self.rf_returns.columns[0]].loc[estimation_range]
                y = asset_returns_reg - rf_returns_reg
                y = y.dropna()
                # if not enough data, skip the ticker
                if y.shape[0] < 2:
                    print(f"Not enough data for ticker {ticker} on event date {event_date}. Skipping.")
                    continue
                y.name = ticker
                y_reg = np.ndarray.flatten(y.to_numpy())
                x = pd.merge(y, market_returns_reg, left_index=True, right_index=True)
                x = x[x.columns[1]]  # Keep only the market returns column
                x = x.to_numpy().reshape(-1,1)

                model = LinearRegression()
                model.fit(x,y_reg)

                # Store results
                betas[event_date].loc["beta", ticker] = model.coef_[0]

        self.betas = betas
        return

    def compute_normal_returns(self):
        """
        Computes the normal returns based on the betas estimated during the regression step.
        :return:
        """
        # To store the normal returns (same data structure as the betas)
        normal_returns = {}
        available_dates = sorted(self.universe_dct.keys())
        for event_date in self.events_dates:
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            closest_date = max([d for d in available_dates if d <= event_date_datetime], default=None)

            if closest_date is not None:
                tickers = self.universe_dct[closest_date]
            else:
                tickers = []

            normal_returns[event_date] = pd.DataFrame(data=np.nan, index=[self.event_windows[event_date]], columns=tickers)

        # Compute the normal returns for each event date and for each stock in the universe.
        # normal returns (time t, stock i) = rf_t + beta_i * market_returns_t
        for event_date in normal_returns.keys():
            event_window = self.event_windows[event_date]
            if len(event_window)==0:
                continue
            betas_i = self.betas[event_date].copy()
            betas_i = pd.concat([betas_i]* len(event_window), ignore_index=False)
            normal_returns_i = self.rf_returns.loc[event_window].values + betas_i.values * self.market_returns.loc[event_window].values
            normal_returns[event_date].loc[event_window] = normal_returns_i

        self.normal_returns = normal_returns
        return

    def compute_abnormal_returns(self):
        """
        Computes the abnormal returns as the difference between the asset returns and the normal returns.
        as in "Firm-Level Exposure to Trade Policy Shocks:
        A Multi-dimensional Measurement Approach" (June 2023) by Bruno, Goltz and Luyten.
        :return:
        """
        # To store the abnormal returns (same data structure as the normal returns)
        abnormal_returns = {}
        available_dates = sorted(self.universe_dct.keys())
        for event_date in self.events_dates:
            event_date_datetime = pd.to_datetime(event_date, format='%Y-%m-%d')
            closest_date = max([d for d in available_dates if d <= event_date_datetime], default=None)

            if closest_date is not None:
                tickers = self.universe_dct[closest_date]
            else:
                tickers = []

            abnormal_returns[event_date] = pd.DataFrame(data=np.nan, index=[self.event_windows[event_date]], columns=tickers)

        # Compute the abnormal returns for each event date and for each stock in the universe.
        for event_date in abnormal_returns.keys():
            event_window = self.event_windows[event_date]
            if len(event_window)==0:
                continue
            tickers = abnormal_returns[event_date].columns.tolist()
            existing_tickers = [t for t in tickers if t in self.asset_returns.columns]

            abnormal_returns_i = self.asset_returns.loc[event_window,existing_tickers].values - self.normal_returns[event_date].loc[event_window, existing_tickers].values
            abnormal_returns[event_date].loc[event_window, existing_tickers] = abnormal_returns_i

        self.abnormal_returns = abnormal_returns
        return

    def compute_cumulative_abnormal_returns(self):
        """
        Computes the cumulative abnormal returns for each event date and across all events.
        :return:
        """
        # To store the cumulative abnormal returns by event
        cumulative_abnormal_returns_by_event = {event_date:pd.DataFrame() for event_date in self.abnormal_returns.keys()}
        for event_date in self.abnormal_returns.keys():
            event_window = self.abnormal_returns[event_date].index.tolist()
            if len(event_window)==0:
                continue

            # Compute the sum of abnormal returns over the event window (across rows)
            summed_returns = self.abnormal_returns[event_date].loc[event_window].sum(skipna=False)
            df = pd.DataFrame([summed_returns.values],  # wrap in list to make 2D
                              index=[event_window[-1]],  # index must be a list
                              columns=self.abnormal_returns[event_date].columns)
            cumulative_abnormal_returns_by_event[event_date] = df

        self.cumulative_abnormal_returns_by_event = cumulative_abnormal_returns_by_event

        # CAUTIOUS ALL STOCKS ARE NOT ALIGNED!
        # To store the cumulative abnormal returns across all events
        common_stocks_across_events = None
        last_event_date = None
        for event_date in cumulative_abnormal_returns_by_event.keys():
            cols = cumulative_abnormal_returns_by_event[event_date].columns.tolist()
            if len(cols)==0:
                continue
            last_event_date = event_date
            if common_stocks_across_events is None:
                common_stocks_across_events = cols
            else:
                common_stocks_across_events = list(set(common_stocks_across_events) & set(cols))

        # Now we can compute the average cumulative abnormal returns across all events
        summed_car = pd.DataFrame(data=np.nan,
                                  index=[last_event_date],
                                  columns=common_stocks_across_events)
        nb_periods = 0
        for event_date in cumulative_abnormal_returns_by_event.keys():
            if cumulative_abnormal_returns_by_event[event_date].empty:
                continue
            nb_periods += 1
            df_single = cumulative_abnormal_returns_by_event[event_date].loc[:, common_stocks_across_events].copy()
            df_single.index = [last_event_date] # because .add() requires same index
            summed_car = summed_car.add(df_single, fill_value=0.0)
        if nb_periods > 0:
            cumulative_abnormal_returns_across_events = summed_car / nb_periods

        self.cumulative_abnormal_returns_across_events = cumulative_abnormal_returns_across_events if nb_periods > 0 else pd.DataFrame()
        return

    def compute_mean_cumulative_abnormal_returns_trade_peace_war_stocks_by_event(self,
                                                                                 mapping_api_to_ticker:dict,
                                                                                 bag_of_words:pd.DataFrame,
                                                                                 condition_column:str = "sum",
                                                                                 bow_or_bowws:str = "bow"
                                                                                 ):

        bow_q = ds.count_companies_talking_at_least_once_about_tariffs_per_quarter(bag_of_words,
                                                                                   condition_column=condition_column,
                                                                                   bow_or_bowws=bow_or_bowws)["bow_q"]
        bow_q["quarter_dt"] = bow_q["quarter"].dt.to_timestamp()
        event_dates_dt = pd.to_datetime(self.events_dates)
        quarters = event_dates_dt.to_period("Q")
        prev_quarters = (quarters - 1)
        prev_quarters_start = prev_quarters.asfreq("Q").start_time

        # To store
        mean_df = pd.DataFrame(data=np.nan, index=pd.to_datetime(self.events_dates), columns=["mean_car_trade_peace", "mean_car_trade_war"])
        # for each event_date / quarter, we get the tickers that signaled at least once
        for i,event_date in enumerate(self.events_dates):
            print(f"Processing event date: {event_date}")
            curr_q = [prev_quarters_start[i]]
            bow_q_on_event = bow_q.loc[bow_q["quarter_dt"].isin(curr_q)]

            if bow_or_bowws == "bow":
                tickers_api_signaled_on_event = bow_q_on_event.loc[bow_q_on_event["signal_once"] > 0.0, "ticker_api"].unique().tolist()
                tickers_api_non_signaled_on_event = bow_q_on_event.loc[bow_q_on_event["signal_once"] == 0.0, "ticker_api"].unique().tolist()
            elif bow_or_bowws == "bowws":
                tickers_api_signaled_on_event = bow_q_on_event.loc[bow_q_on_event["trade_sentiment"] > 0.0, "ticker_api"].unique().tolist()
                tickers_api_non_signaled_on_event = bow_q_on_event.loc[bow_q_on_event["trade_sentiment"] < 0.0, "ticker_api"].unique().tolist()
            else:
                raise ValueError("bow_or_bowws must be either 'bow' or 'bowws'.")

            tickers_signaled_on_event = [mapping_api_to_ticker[ticker_api] for ticker_api in tickers_api_signaled_on_event]
            tickers_non_signaled_on_event = [mapping_api_to_ticker[ticker_api] for ticker_api in tickers_api_non_signaled_on_event]
            if len(tickers_signaled_on_event) == 0 or len(tickers_non_signaled_on_event) == 0:
                print(f"No tickers signaled or non-signaled on event date {event_date}. Skipping.")
                continue

            df = self.cumulative_abnormal_returns_by_event[event_date]
            mean_trade_peace = df.loc[:, df.columns.isin(tickers_signaled_on_event)].mean().mean()
            mean_trade_war = df.loc[:, df.columns.isin(tickers_non_signaled_on_event)].mean().mean()
            mean_df.loc[event_date, "mean_car_trade_peace"] = mean_trade_peace
            mean_df.loc[event_date, "mean_car_trade_war"] = mean_trade_war

        # self.mean_cumulative_abnormal_returns_trade_peace_war_by_event = mean_df
        return mean_df