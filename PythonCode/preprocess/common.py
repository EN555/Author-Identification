import os

import nltk.tokenize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PythonCode.Constants import *
from pathlib import Path
from itertools import islice


# can be passed to pipeline using data_filter parameter


def chunking(df: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
    """
    split each entry into shorter texts
    @param chunk_size: number of words in each new entry
    """
    rows = []
    for _, row in df.iterrows():
        words = row[TEXT_COLUMN_LABEL].split(' ')
        chunks = [' '.join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
        for chunk in chunks:
            tmp_row = row.copy()
            tmp_row[TEXT_COLUMN_LABEL] = chunk
            rows.append(tmp_row.copy())
    return pd.DataFrame(rows)


def num_sentences_based_chucking(df: pd.DataFrame, chunk_size: int):
    def create_chunk(curr_chunk, row):
        tmp_row = row.copy()
        tmp_row[TEXT_COLUMN_LABEL] = "".join(curr_chunk)
        return tmp_row.copy()

    rows = []
    for _, row in df.iterrows():
        sentences = nltk.tokenize.sent_tokenize(row[TEXT_COLUMN_LABEL])
        curr_chunk = []
        for sentence in sentences:
            curr_chunk.append(sentence)
            if len(curr_chunk) == chunk_size:
                rows.append(create_chunk(curr_chunk.copy(), row))
                curr_chunk = []
        rows.append(create_chunk(curr_chunk, row))  # add the last one
    return pd.DataFrame(rows)


def combine_features(feature_extractors: list, x_train: pd.DataFrame, x_test: pd.DataFrame) -> (
        pd.DataFrame, pd.DataFrame):
    """
    @param feature_extractors list of feature extractor callback like complex_style_features_extraction
    """
    train_results, test_results = [], []
    for feature_extractor in feature_extractors:
        out_train, out_test = feature_extractor(x_train, x_test)
        train_results.append(out_train)
        test_results.append(out_test)
    return pd.concat([df.reset_index(drop=True) for df in train_results], axis=1), \
           pd.concat([df.reset_index(drop=True) for df in test_results], axis=1)


def preprocess_pipeline(data_path: str, number_of_authors: int, repesention_handler, normalize: bool,
                        scaler=StandardScaler(), test_size: float = 0.3, random_state=1,
                        save_path="../../Data/clean/", data_filter=None, cache=False, std_thr: float = 0.3,
                        **kwargs) -> (
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    if cache:
        return pd.read_csv(os.path.join(save_path, "x_train_clean.csv")), \
               pd.read_csv(os.path.join(save_path, "x_test_clean.csv")), \
               pd.read_csv(os.path.join(save_path, "y_train_clean.csv")), \
               pd.read_csv(os.path.join(save_path, "y_test_clean.csv"))

    df = load_data(data_path, number_of_authors)  # train and validation

    if data_filter is not None:
        df = data_filter(df)

    X, Y = pd.DataFrame(df[TEXT_COLUMN_LABEL], columns=[TEXT_COLUMN_LABEL]), df[AUTHOR_NAME_COLUMN_NAME]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    x_train, x_test = repesention_handler(x_train, x_test, **kwargs)

    # remove too large values
    x_train[x_train > 2 ** 60] = 2 ** 60
    x_test[x_test > 2 ** 60] = 2 ** 60

    if normalize:
        scaler.fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]

    # turn to correct format
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # remove low variance features
    keep_indexes = x_train.std() > std_thr
    x_train = x_train.loc[:, keep_indexes]
    x_test = x_test.loc[:, keep_indexes]

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        x_train.to_csv(os.path.join(save_path, "x_train.csv"))
        y_train.to_csv(os.path.join(save_path, "y_train.csv"))
        x_test.to_csv(os.path.join(save_path, "x_test.csv"))
        y_test.to_csv(os.path.join(save_path, "y_test.csv"))

    return x_train, x_test, y_train, y_test


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
                curr_row[TEXT_COLUMN_LABEL] = book.read()
            rows_list.append(curr_row.copy())
    return pd.DataFrame(rows_list)


def merge_datasets(data_path: str = "../Data/C50") -> pd.DataFrame:
    df_test = load_data(f"{data_path}/C50test", 50)
    df_train = load_data(f"{data_path}/C50train", 50)
    return df_train.append(df_test, ignore_index=True)
