import numpy as np
import pandas as pd
import gzip
import pickle
import os
from src.data_earnings_calls_transcripts import DataEarningsCallsTranscripts
from src.nlp_models import BagOfWords, BagOfWordsWithSentiment, CustomModels
from sklearn.linear_model import LogisticRegression

# if __name__ == "__main__":
####################################################### INPUTS #####################################################
download_or_load_transcripts = "load" # the transcripts can be downloaded from the API or loaded from a local file. We
# recommend to always load the transcripts from a local file as the download from the API is limited to a number of calls
# per month plus it's very long (~17 hours for the russell 1000 between 2008-2025).
index_constituents_feather_path = os.path.join("data", "RIY Index constituents.feather")
####################################################################################################################
# Step 1 - Data Loading and Preprocessing
dect = DataEarningsCallsTranscripts(index_constituents_feather_path=index_constituents_feather_path)
dect.load_index_constituents()
if download_or_load_transcripts == "download":
    dect.get_earnings_transcript_for_index_constituents()
    dect.preprocess_transcripts()
elif download_or_load_transcripts == "load":
    files = {
        "formatted_transcripts": os.path.join("data", "formatted_transcripts_gzip.pkl.gz"),
        "formatted_transcripts_preprocessed": os.path.join("data", "formatted_transcripts_preprocessed_gzip.pkl.gz")
    }
    for key, path in files.items():
        print(f"Loading {key} from {path}...")
        with gzip.open(path, "rb") as handle:
            setattr(dect, key, pickle.load(handle))
else:
    raise ValueError("download_or_load_transcripts must be either 'download' or 'load'.")

# Step 1 - Sentences of interest retrieval
cm = CustomModels(formatted_transcripts=dect.formatted_transcripts,
                  start_date_training="2007-05-10",
                  end_date_training="2019-01-01",
                  start_date_validation = "2019-01-02",
                  end_date_validation = "2024-01-01",
                  start_date_test = "2024-01-02",
                  end_date_test = "2025-07-24"
                  )
cm.apply_retrieve_sentences_fast() # takes c. 1min
cm.get_unlabelled_data_flat()
# fwd looking bias?no because even if the model saws our sentences, this is the training data?
# No as we'll label the whole dataset. Can be good but true oos forecast must be after the model
# date release?
# cm.zero_shot_classification_sentence_level(unlabelled_train_val_data_flat=cm.unlabelled_data_flat.loc[cm.start_date_training:cm.end_date_validation],
#                                            model="facebook/bart-large-mnli",
#                                            hypothesis_template= "This statement is positive {}.",
#                                            save_in_self_sentence_level_labels=True,
#                                            return_results=False
#                                            ) # to optimize, takes c. 25min
# if you want to save time and directly load the results from the above function,
# de-comment these lines
with gzip.open(os.path.join("data", "zero_shot_classification_sentence_level_labels_train_val.pkl.gz"), "rb") as handle:
    cm.sentence_level_labels = pickle.load(handle)
cm.get_accuracy_zero_shot_classification(human_label_df=None,
                                         loading_path_human=os.path.join("outputs", "zero_shot_classification_results_human_label.xlsx"),
                                         file_extension="xlsx",
                                         usecols="A:H",
                                         threshold_range=(0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
                                         )
# note: the optimal threshold is generally the highest (0.9) with an accuracy of 1.0 but very small sample size
# (4 out of 50).
cm.get_zsc_sentence_level_label_filtered_threshold(threshold=cm.optimal_threshold_zsc)
cm.split_train_validation_test()
clf = LogisticRegression(max_iter=1000,
                         solver="lbfgs",
                         class_weight="balanced")  # lbfgs supports multinomial
# error below, need to fix!
cm.fit_model(LogisticRegression,
             max_iter=1000,
             solver='lbfgs',
             class_weight='balanced')
# Last step: zero_shot_classification_transcript_level_aggregation
# [to be completed]












# # Step 3: train a classification model
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# # Create TF-IDF vectors
# vectorizer = TfidfVectorizer(
#     max_features=10000,    # Top 10K most frequent words
#     ngram_range=(1, 2),    # Use 1-grams and 2-grams
#     stop_words='english'   # Remove common words
# )
# # non_unlabelled_sentence_level_labels = self.sentence_level_label_filtered_threshold
# X = vectorizer.fit_transform(non_unlabelled_sentence_level_labels["sentence"])
# y = non_unlabelled_sentence_level_labels["labels"]
# # DO NOT TAKE RANDOM SPLIT, TAKE CHRONOLOGICAL DUE TO MODEL USED FOR LABELLING
# start_date_training="2007-05-10"
# end_date_training="2019-01-01"
# # validation is used to determine the best tariff-sentiment classification model (logreg, RF)
# start_date_validation = "2019-01-02"
# end_date_validation = "2024-01-01"
# # OOS test - oos test is assessed with the trade war vs trade peace stocks difference returns
# start_date_test = "2024-01-02"
# end_date_test = "2025-07-24"
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train = cm.sentence_level_label_filtered_threshold.loc[start_date_training:end_date_training, "sentence"]
# y_train = cm.sentence_level_label_filtered_threshold.loc[start_date_training:end_date_training, "labels"]
# X_validation = cm.sentence_level_label_filtered_threshold.loc[start_date_validation:end_date_validation, "sentence"]
# y_validation = cm.sentence_level_label_filtered_threshold.loc[start_date_validation:end_date_validation, "labels"]
# X_test = cm.sentence_level_label_filtered_threshold.loc[start_date_test:end_date_test, "sentence"]
# y_test = cm.sentence_level_label_filtered_threshold.loc[start_date_test:end_date_test, "labels"]


# clf.fit(X_train, y_train)
# print("Accuracy:", clf.score(X_test, y_test))
