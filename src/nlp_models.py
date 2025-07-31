import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from .transform import TradePolicyShockSentimentAnalyzer
from joblib import Parallel, delayed
from src.utilities import retrieve_sentences_fast, get_unlabelled_data_flat_util, get_human_machine_accuracy_classification
from tqdm import tqdm
from transformers import pipeline
import torch
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer

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


class CustomModels:
    def __init__(self,
                 formatted_transcripts:pd.DataFrame,
                 keywords:Tuple[str]=(
                         'tariff', 'tariffs', 'import duty', 'import duties', 'import barrier', 'import barriers', 'import ban', 'import bans', 'import tax', 'import taxes',
                         'import subsidy', 'import subsidies', 'export ban', 'export bans', 'export tax', 'export taxes', 'export subsidy', 'export subsidies',
                         'government subsidy', 'government subsidies', 'GATT', 'WTO', 'World Trade Organization',
                         'trade treaty', 'trade treaties', 'trade agreement', 'trade agreements', 'trade policy', 'trade policies', 'trade act', 'trade acts',
                         'trade relationship', 'trade relationships', 'free trade', 'free trades', 'Doha round', 'Doha rounds', 'Uruguay round', 'Uruguay rounds',
                         'dumping', 'border tax', 'border taxes'),
                 labels:Tuple[str]=("positive", "negative", "neutral"),
                 start_date_training:str=None,
                 end_date_training:str=None,
                 start_date_validation:str=None,
                 end_date_validation:str=None,
                 start_date_test:str=None,
                 end_date_test:str=None
                 ):
        self.formatted_transcripts = formatted_transcripts
        self.index_names = list(self.formatted_transcripts.index.names)
        self.keywords = keywords
        self.keywords_lower = [k.lower() for k in self.keywords]
        self.labels = list(labels)

        self.start_date_training = start_date_training
        self.end_date_training = end_date_training
        self.start_date_validation =start_date_validation
        self.end_date_validation = end_date_validation
        self.start_date_test= start_date_test
        self.end_date_test = end_date_test

        self.sentences_of_interest_df = None
        self.unlabelled_data_flat = None
        self.sentence_level_labels = None
        self.accuracy_zero_shot_classification = None
        self.optimal_threshold_zsc = None
        self.sentence_level_label_filtered_threshold = None

        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
        self.clf = None

        self.transcript_scores = None
        self.transcript_labels = None

    def apply_retrieve_sentences_fast(self):
        tqdm.pandas()
        formatted_transcripts_series = self.formatted_transcripts["transcript"]
        results = Parallel(n_jobs=-1)(
            delayed(retrieve_sentences_fast)(text, self.keywords_lower)
            for text in tqdm(formatted_transcripts_series, desc="Extraction", unit="doc")
        )

        sentences_of_interest_series = pd.Series(results, index=formatted_transcripts_series.index)
        self.sentences_of_interest_df = self.formatted_transcripts.assign(
            sentences_of_interest=sentences_of_interest_series
        ).drop(columns=["transcript"])

    def get_unlabelled_data_flat(self,
                                 start_date:str=None,
                                 end_date:str=None
                                 ):
        if start_date is None:
            self.sentences_of_interest_df.index.get_level_values('filing_date').min()
        if end_date is None:
            self.sentences_of_interest_df.index.get_level_values('filing_date').max()

        res = get_unlabelled_data_flat_util(sentences_of_interest_df=self.sentences_of_interest_df,
                                            start_date=start_date,
                                            end_date=end_date,
                                            keywords_lower=self.keywords_lower)
        if self.unlabelled_data_flat is None:
            self.unlabelled_data_flat = res

    def zero_shot_classification_sentence_level(self,
                                                unlabelled_train_val_data_flat: pd.DataFrame,
                                                model:str="facebook/bart-large-mnli",
                                                hypothesis_template:str="This statement is positive {}.",
                                                saving_path:str=None,
                                                save_in_self_sentence_level_labels:bool=True,
                                                return_results:bool=False
                                                ):
        device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
        classifier = pipeline("zero-shot-classification",
                              model=model,
                              device=device)
        results = pd.DataFrame(data=np.nan,
                               index=unlabelled_train_val_data_flat.index,
                               columns=["sentence"] + self.labels,
                               dtype=object
                               )
        for idx,sent in tqdm(zip(unlabelled_train_val_data_flat.index, unlabelled_train_val_data_flat["sentences_of_interest"]),
                             total=unlabelled_train_val_data_flat.shape[0],
                             desc="Zero-shot classification"):
            res = classifier(sent,
                             candidate_labels=self.labels,
                             hypothesis_template=hypothesis_template,
                             multi_label=False
                             )
            results.loc[idx, "sentence"] = sent
            dct_scores = dict(zip(res["labels"], res["scores"]))
            for label in self.labels:
                results.loc[idx, label] = dct_scores.get(label, np.nan)

        if saving_path is not None:
            results.to_excel(saving_path, index=True)

        if save_in_self_sentence_level_labels:
            self.sentence_level_labels = results
        if return_results:
            return results

    def get_accuracy_zero_shot_classification(self,
                                              human_label_df=None,
                                              loading_path_human:str=r'.\outputs\zero_shot_classification_results_human_label.xlsx',
                                              file_extension:str='xlsx',
                                              usecols:str="A:H",
                                              threshold_range:Tuple[float,...]=(0.33,0.4,0.5,0.6,0.7,0.8,0.9)
                                              ):
        res = get_human_machine_accuracy_classification(machine_label_df=self.sentence_level_labels,
                                                        human_label_df=human_label_df,
                                                        loading_path_human=loading_path_human,
                                                        file_extension=file_extension,
                                                        usecols=usecols,
                                                        threshold_range=threshold_range,
                                                        labels=tuple(self.labels))
        self.accuracy_zero_shot_classification = res
        self.optimal_threshold_zsc = max(res, key=lambda k: res[k]['accuracy'])
        if isinstance(self.optimal_threshold_zsc, str):
            self.sentence_level_labels["labels"] = self.sentence_level_labels[list(self.labels)].apply(lambda x: x.idxmax(), axis=1)
        else:
            self.sentence_level_labels["labels"] = self.sentence_level_labels[list(self.labels)].apply(
                lambda x: x.idxmax() if x.max() >= self.optimal_threshold_zsc else "unlabelled", axis=1)

    def get_zsc_sentence_level_label_filtered_threshold(self,
                                                        threshold:float):

        # Now, add "unlabelled" to the labels according to a threshold
        self.sentence_level_labels["labels_filtered"] = self.sentence_level_labels[self.labels].apply(
            lambda x: x.idxmax() if
            x.max() > threshold else "unlabelled", axis=1
        )
        # Now filter on the non-unlabelled labels
        sentence_level_label_filtered_threshold = self.sentence_level_labels[["sentence", "labels"]].loc[
            self.sentence_level_labels["labels_filtered"] != "unlabelled"]
        print(f"shape of sentence_level_label_filtered_threshold:{sentence_level_label_filtered_threshold.shape}")
        for label in self.labels:
            print(f"freq of {label} labels{(sentence_level_label_filtered_threshold['labels'] == label).sum() / sentence_level_label_filtered_threshold.shape[0]} ({(sentence_level_label_filtered_threshold['labels'] == label).sum()})")

        self.sentence_level_label_filtered_threshold = sentence_level_label_filtered_threshold

    def split_train_validation_test(self):
        self.x_train = self.sentence_level_label_filtered_threshold.loc[self.start_date_training:self.end_date_training, "sentence"]
        self.y_train = self.sentence_level_label_filtered_threshold.loc[self.start_date_training:self.end_date_training, "labels"]
        self.x_validation = self.sentence_level_label_filtered_threshold.loc[self.start_date_validation:self.end_date_validation,
                       "sentence"]
        self.y_validation = self.sentence_level_label_filtered_threshold.loc[self.start_date_validation:self.end_date_validation,
                       "labels"]
        self.x_test = self.sentence_level_label_filtered_threshold.loc[self.start_date_test:self.end_date_test, "sentence"]
        self.y_test = self.sentence_level_label_filtered_threshold.loc[self.start_date_test:self.end_date_test, "labels"]

    def fit_model(self, model_class: type[ClassifierMixin], **kwargs):
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data not assigned")
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test data not assigned")

        vectorizer = TfidfVectorizer(
            max_features=10000,  # Top 10K most frequent words
            ngram_range=(1, 2),  # Use 1-grams and 2-grams
            stop_words='english'  # Remove common words
        )
        sentences = self.sentence_level_label_filtered_threshold["sentence"]
        x = vectorizer.fit_transform(sentences)

        date_index = self.sentence_level_label_filtered_threshold.index.get_level_values("filing_date")
        mask_train = (date_index >= self.start_date_training) & (date_index <= self.end_date_training)
        mask_test = (date_index >= self.start_date_test) & (date_index <= self.end_date_test)

        # Use the mask to slice the sparse matrix by integer positions
        x_train = x[mask_train]
        x_test = x[mask_test]
        y_train = self.y_train[mask_train]
        y_test = self.y_test[mask_test]

        # Fit and evaluate
        self.clf = model_class(**kwargs)
        self.clf.fit(x_train, y_train)
        print("Accuracy:", self.clf.score(x_test, y_test))

    def zero_shot_classification_transcript_level_aggregation(self,
                                                              threshold:float=0.5
                                                              ):
        transcript_scores = self.sentence_level_labels.groupby(level=self.index_names)[self.labels].mean()
        transcript_scores['max_score'] = transcript_scores.max(axis=1)
        transcript_scores['label'] = transcript_scores[self.labels].idxmax(axis=1)
        transcript_scores.loc[transcript_scores['max_score'] < threshold, 'label'] = 'unlabelled'
        if self.transcript_scores is None:
            self.transcript_scores = transcript_scores
        transcript_labels = transcript_scores[['label']]
        if self.transcript_labels is None:
            self.transcript_labels = transcript_labels
