import numpy as np
import nltk
import pandas as pd
import gensim.downloader
import gensim
import re
from typing import Optional

from PythonCode.preprocess.preprocess import load_data
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GRU, AvgPool1D
from keras.models import Sequential

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
print("downloading pretrained embedding model.\nthis may take a while...")
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
EMBEDDING_SIZE = 50
NUM_OF_SENTENCE_CHUNK = 3
MAX_LENGTH = 170
TEST_PART = 0.1


def tranform_word(word: str) -> Optional[np.ndarray]:
    word = re.sub(r'[^a-z]', '', word.lower())
    if word in glove_vectors:
        return glove_vectors[word]
    return None


def complex_tranform_word(word: str):
    result = tranform_word(word)
    if result is None:
        token = lemmatizer.lemmatize(word)
        if token in glove_vectors:
            return glove_vectors[token]
        token = stemmer.stem(word)
        if token in glove_vectors:
            return glove_vectors[token]
    return result


def num_sentences_based_chucking(df: pd.DataFrame, chunk_size: int):
    rows = []
    for row in df:
        sentences = nltk.tokenize.sent_tokenize(row)
        curr_chunk = []
        for sentence in sentences:
            curr_chunk.append(sentence)
            if len(curr_chunk) == chunk_size:
                rows.append("".join(curr_chunk))
                curr_chunk = []

        rows.append("".join(curr_chunk))  # add the last one
    return pd.DataFrame(rows)


def pad_matrix(arr: np.ndarray, max_length: int) -> Optional[np.ndarray]:
    if arr.size == 0:
        return None
    if arr.shape[0] == max_length:
        return arr
    if arr.shape[0] > max_length:
        return arr[:max_length, :]
    return np.concatenate([arr, np.zeros((max_length - arr.shape[0], arr.shape[1]))], axis=0, dtype=float)


def sentence_level_preprocess_helper(text: str):
    words = nltk.word_tokenize(text)
    result = []
    missing_embedding_count = 0
    for word in words:
        embedding = tranform_word(word)
        if embedding is not None:
            result.append(embedding)
        else:
            missing_embedding_count += 1
    return pd.Series(
        {"data": pad_matrix(np.array(result), MAX_LENGTH), "missing_embedding_count": missing_embedding_count})


def get_datasets(data_path: str = "../Data/C50") -> pd.DataFrame:
    df_test = load_data(f"{data_path}/C50test", 50)
    df_train = load_data(f"{data_path}/C50train", 50)
    return df_train.append(df_test, ignore_index=True)


def preprocess_labels(y: pd.Series) -> np.ndarray:
    return pd.Categorical(y).codes


def sentence_level_preprocess(df: pd.DataFrame):
    def helper(X):
        X = num_sentences_based_chucking(X, NUM_OF_SENTENCE_CHUNK)[0]
        res = X.swifter.apply(sentence_level_preprocess_helper)["data"].dropna().reset_index(drop=True)
        return np.vstack(res).reshape((res.size, MAX_LENGTH, EMBEDDING_SIZE))

    X_train, X_test, y_train, y_test = train_test_split(df["book_text"], df["author_name"], test_size=TEST_PART)
    return helper(X_train), helper(X_test), preprocess_labels(y_train), preprocess_labels(y_test)


MAX_SENTENCE_LENGTH = 70
MAX_NUMBER_OF_SENTENCE = 45


def pad_array(arr: np.ndarray, pad_size: int):  # TODO: reuse pad_matrix instead
    if arr.size == pad_size:
        return arr
    elif arr.size > pad_size:
        return arr[:pad_size, ]
    return np.concatenate([arr, np.zeros(pad_size - arr.size)], dtype=float)


def article_level_preprocess_helper(text: str):
    sentences = nltk.sent_tokenize(text)
    result = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        curr_result = []
        for word in words:
            embedding = tranform_word(word)
            if embedding is not None:
                curr_result.append(embedding)
        if len(curr_result) != 0:
            result.append(pad_array(np.array(curr_result, dtype=float).mean(axis=1, dtype=float), MAX_SENTENCE_LENGTH))
    return pad_matrix(np.array(result), MAX_NUMBER_OF_SENTENCE)


def article_level_preprocess(df: pd.DataFrame):
    def helper(X):
        res = X.swifter.apply(article_level_preprocess_helper).reset_index(drop=True)
        return np.vstack(res).reshape((res.size, MAX_NUMBER_OF_SENTENCE, MAX_SENTENCE_LENGTH))

    X_train, X_test, y_train, y_test = train_test_split(df["book_text"], df["author_name"], test_size=TEST_PART)
    return helper(X_train), helper(X_test), preprocess_labels(y_train), preprocess_labels(y_test)


def model_sentence_level():
    model = Sequential()
    model.add(GRU(100, recurrent_dropout=0.2, input_shape=(MAX_LENGTH, EMBEDDING_SIZE), return_sequences=True))
    model.add(AvgPool1D(pool_size=(170,)))
    model.add(Dense(50, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()


if __name__ == '__main__':
    print("im here")
