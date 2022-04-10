import os
import nltk.tokenize
import pandas as pd
from src.Constants import *
from itertools import islice


def chunking(df: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
    """
    split each entry into shorter texts
    @param chunk_size: number of words in each new entry
    @param df
    """
    rows = []
    for _, row in df.iterrows():
        words = row[TEXT_COLUMN_NAME].split(' ')
        chunks = [' '.join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
        for chunk in chunks:
            tmp_row = row.copy()
            tmp_row[TEXT_COLUMN_NAME] = chunk
            rows.append(tmp_row.copy())
    return pd.DataFrame(rows)


def num_sentences_based_chucking(df: pd.DataFrame, chunk_size: int):
    def create_chunk(curr_chunk, row):
        tmp_row = row.copy()
        tmp_row[TEXT_COLUMN_NAME] = "".join(curr_chunk)
        return tmp_row.copy()

    rows = []
    for _, row in df.iterrows():
        sentences = nltk.tokenize.sent_tokenize(row[TEXT_COLUMN_NAME])
        curr_chunk = []
        for sentence in sentences:
            curr_chunk.append(sentence)
            if len(curr_chunk) == chunk_size:
                rows.append(create_chunk(curr_chunk.copy(), row))
                curr_chunk = []
        rows.append(create_chunk(curr_chunk, row))  # add the last one
    return pd.DataFrame(rows)


def remove_low_std_features(x_train,x_test,std_thr=0.3):
    keep_indexes = x_train.std() > std_thr
    return x_train.loc[:, keep_indexes], x_test.loc[:, keep_indexes]


def remove_too_large_values(x_train,x_test):
    x_train[x_train > 2 ** 60] = 2 ** 60
    x_test[x_test > 2 ** 60] = 2 ** 60
    return x_train,x_test


def load_data(path: str, number_of_authors: int) -> pd.DataFrame:
    rows_list = []
    _, authors, _ = next(os.walk(path))
    for author_name in islice(authors, number_of_authors):
        curr_row = {AUTHOR_NAME_COLUMN_NAME: author_name}
        author_path = os.path.join(path, author_name)
        _, _, books_files = next(os.walk(author_path))
        for book_name in books_files:
            curr_row[BOOK_NAME_COLUMN_NAME] = book_name
            with open(os.path.join(author_path, book_name), "r") as book:
                curr_row[TEXT_COLUMN_NAME] = book.read()
            rows_list.append(curr_row.copy())
    return pd.DataFrame(rows_list)


def merge_datasets(data_path: str = "../data/C50") -> pd.DataFrame:
    df_test = load_data(f"{data_path}/C50test", 50)
    df_train = load_data(f"{data_path}/C50train", 50)
    return df_train.append(df_test, ignore_index=True)



def combine_features(feature_extractors: list, x_train: pd.DataFrame, x_test: pd.DataFrame) -> (
        pd.DataFrame, pd.DataFrame):
    """
    @param feature_extractors list of feature extractor callback like complex_style_features_extraction
    @param x_train
    @param x_test
    """
    train_results, test_results = [], []
    for feature_extractor in feature_extractors:
        out_train, out_test = feature_extractor(x_train, x_test)
        train_results.append(out_train)
        test_results.append(out_test)
    return pd.concat([df.reset_index(drop=True) for df in train_results], axis=1), \
           pd.concat([df.reset_index(drop=True) for df in test_results], axis=1)
