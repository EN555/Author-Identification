import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PythonCode.Constants import *
from pathlib import Path
from itertools import islice


def filter_for_base_case():
    """
    filter two authors
    """
    pass


# can be passed to pipeline using data_filter parameter
def chunking(df: pd.Dataset, chunk_size: int) -> pd.Dataset:
    """
    split each entry into shorter texts
    @param chunk_size: number of words in eac new entry
    """
    rows = []
    for _, row in df.iterrows():
        words = row[TEXT_COLUMN_LABEL].split(' ')
        chunks = [' '.join(words[i, i + chunk_size]) for i in range(0, len(words), chunk_size)]
        for chunk in chunks:
            tmp_row = row.copy()
            tmp_row[TEXT_COLUMN_LABEL] = chunk
            rows.append(tmp_row.copy())
    return pd.DataFrame(rows)


def preprocess_pipeline(data_path: str, number_of_authors: int, repesention_handler, normalize: bool,
                        scaler=StandardScaler(), test_size: float = 0.3, random_state=1,
                        save_path="../../Data/clean/", data_filter=None, cache=False, **kwargs) -> (
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    if cache:
        # TODO: extract y_test_clean...
        return pd.read_csv(os.path.join(data_path, "x_train_clean.csv")), \
               pd.read_csv(os.path.join(data_path, "x_test_clean.csv")), \
               pd.read_csv(os.path.join(data_path, "y_train_clean.csv")), \
               pd.read_csv(os.path.join(data_path, "y_test_clean.csv"))

    df = load_data(data_path, number_of_authors)  # train and validation

    if data_filter is not None:
        df = data_filter(df)

    X, Y = pd.DataFrame(df[TEXT_COLUMN_LABEL], columns=[TEXT_COLUMN_LABEL]), df[AUTHOR_NAME_COLUMN_NAME]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    x_train, x_test = repesention_handler(x_train, x_test, **kwargs)

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
