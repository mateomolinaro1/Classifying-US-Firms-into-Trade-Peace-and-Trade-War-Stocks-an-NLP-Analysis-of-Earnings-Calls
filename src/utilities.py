import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from typing import Tuple, List
import re

def find_values_by_key(d, target_key):
    """Recursively find all values for a given key in a nested structure."""
    found = []

    if isinstance(d, dict):
        for key, value in d.items():
            if key == target_key:
                found.append(value)
            found += find_values_by_key(value, target_key)
    elif isinstance(d, list):
        for item in d:
            found += find_values_by_key(item, target_key)

    return found


def preprocess_text(df: pd.DataFrame,
                    column_name_to_clean: str = 'transcript',
                    new_column_name: str = 'transcript_cleaned') -> pd.DataFrame:
    """
    Preprocesses the transcripts in the DataFrame by tokenizing, removing stopwords, and stemming.

    :param df: DataFrame containing a 'transcript' column.
    :param column_name_to_clean: The name of the column containing the transcripts to clean.
    :param new_column_name: The name of the new column to store the cleaned transcripts.
    :return: DataFrame with an additional 'transcript_clean' column.
    """
    nltk.download('stopwords')

    # Initialize NLP tools
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")

    def preprocess(text):
        tokens = tokenizer.tokenize(text.lower())
        filtered = [w for w in tokens if w not in stop_words]
        stemmed = [stemmer.stem(w) for w in filtered]
        return " ".join(stemmed)

    dfc = df.copy()
    dfc[new_column_name] = dfc[column_name_to_clean].apply(preprocess)
    dfc = dfc.drop(columns=[column_name_to_clean])
    return dfc


def retrieve_sentences_fast(text:str, keywords_lower:Tuple[str]) -> list:
    sentences = re.split(r'[.?!]', text)
    return [
        s.strip() for s in sentences
        if any(k in s.lower() for k in keywords_lower)
    ]

def get_unlabelled_data_flat_util(sentences_of_interest_df:pd.DataFrame,
                                  start_date:str,
                                  end_date:str,
                                  keywords_lower:List[str] = None
                                  )-> pd.DataFrame:
    unlabelled_data = sentences_of_interest_df.loc[start_date:end_date]
    unlabelled_data_flat = (
        unlabelled_data["sentences_of_interest"]
        .explode()
        .dropna()
        .to_frame(name="sentences_of_interest")
    )
    unlabelled_data_flat['sentence_number'] = (
        unlabelled_data_flat
        .groupby(level=[0, 1])
        .cumcount() + 1
    )
    unlabelled_data_flat = unlabelled_data_flat.set_index('sentence_number', append=True)
    # Double check: deleting sentences that do not contain trade keywords
    if keywords_lower:
        pattern = r"|".join(r"\b" + re.escape(k) + r"\b" for k in keywords_lower)
        mask = unlabelled_data_flat["sentences_of_interest"].str.lower().str.contains(pattern)
        unlabelled_data_flat = unlabelled_data_flat[mask]
    return unlabelled_data_flat

def get_human_machine_accuracy_classification(machine_label_df:pd.DataFrame,
                                              human_label_df:pd.DataFrame=None,
                                              loading_path_human:str=None,
                                              file_extension:str = "xlsx",
                                              usecols:str="A:H",
                                              threshold_range:Tuple[float,...] = (0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                              labels:Tuple[str,...]=("positive", "negative", "neutral")
                                              ) -> dict:
    if loading_path_human is not None:
        if file_extension=='xlsx':
            human_label_df = pd.read_excel(loading_path_human, usecols=usecols)
            human_label_df[['filing_date', 'ticker_api']] = human_label_df[['filing_date', 'ticker_api']].ffill()
            # Set MultiIndex
            human_label_df = human_label_df.set_index(['filing_date', 'ticker_api', 'sentence_number'])
        else:
            raise ValueError("Unsupported file extension. Use 'xlsx' for Excel files.")

    merged = pd.merge(machine_label_df, human_label_df, left_index=True, right_index=True, suffixes=('', '_human'))
    threshold_range_list = ["no_threshold"] +  list(threshold_range)
    accuracy_dict = dict(
        zip(
            threshold_range_list,
            [{'accuracy': None, 'nb_instances': None, 'prop_instances': None} for _ in range(len(threshold_range_list))]
        )
    )

    merged['labels'] = merged[list(labels)].apply(lambda x: x.idxmax(), axis=1)
    for k in accuracy_dict.keys():
        if k == "no_threshold":
            accuracy_dict[k]['accuracy'] = (merged['labels'] == merged['human_label']).mean()
            accuracy_dict[k]['nb_instances'] = merged.shape[0]
            accuracy_dict[k]['prop_instances'] = 1.0
        else:
            merged['labels_filtered'] = merged[list(labels)].apply(lambda x: x.idxmax() if max(x) >= k else 'unlabelled', axis=1)
            if not merged.empty:
                mask = merged['labels_filtered'] != 'unlabelled'
                accuracy_dict[k]['accuracy'] = (merged[mask]['labels_filtered'] == merged[mask]['human_label']).mean()
                accuracy_dict[k]['nb_instances'] = merged[mask].shape[0]
                accuracy_dict[k]['prop_instances'] = merged[mask].shape[0] / merged.shape[0]
    print(accuracy_dict)
    return accuracy_dict
