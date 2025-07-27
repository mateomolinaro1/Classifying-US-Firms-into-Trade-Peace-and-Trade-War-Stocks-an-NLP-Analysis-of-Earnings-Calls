import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

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
