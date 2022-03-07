import pickle

import numpy as np
import nltk
import pandas as pd
import gensim.downloader
import gensim
import re
from typing import Optional
from PythonCode.preprocess.preprocess import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GRU, AvgPool2D
import tensorflow as tf
from tensorflow.keras.models import Sequential

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
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


def sentence_level_preprocess(text: str):
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


def preprocess_helper(X):
    X = num_sentences_based_chucking(X, NUM_OF_SENTENCE_CHUNK)[0]
    res = X.swifter.apply(sentence_level_preprocess)["data"].dropna().reset_index(drop=True)
    return np.vstack(res).reshape((res.size, MAX_LENGTH, EMBEDDING_SIZE))


def preprocess(data_path: str = "../Data/C50"):
    df_test = load_data(f"{data_path}/C50test", 50)
    df_train = load_data(f"{data_path}/C50train", 50)
    df = df_train.append(df_test, ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(df["book_text"], df["author_name"], test_size=TEST_PART)
    X_train = preprocess_helper(X_train)
    X_test = preprocess_helper(X_test)
    return X_train, X_test, y_train, y_test


def model_sentence_level():
    model = Sequential()
    model.add(GRU(100, recurrent_dropout=0.2, input_shape=(MAX_LENGTH, EMBEDDING_SIZE)))
    model.add(AvgPool2D((1, 50)))
    model.add(Dense(50, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    print(model.summary())

