import tensorflow as tf
import numpy as np
import nltk
import pandas as pd
import gensim.downloader
import gensim
import re
from typing import Optional
import swifter
from PythonCode.Constants import *
from PythonCode.Constants import TEXT_COLUMN_LABEL
from sklearn.model_selection import train_test_split

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
print("downloading pretrained embedding model.\nthis may take a while...")
glove_vectors = gensim.downloader.load(GLOVE_MODEL_NAME)

DATA_PATH = "../../Data/C50"
OUTPUTS_PATH = "."


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
    def create_chunk():
        temp_row = row.copy()
        temp_row[TEXT_COLUMN_LABEL] = "".join(curr_chunk)
        rows.append(temp_row)
        return temp_row

    rows = []
    for _, row in df.iterrows():
        sentences = nltk.tokenize.sent_tokenize(row["X"])
        curr_chunk = []
        for sentence in sentences:
            curr_chunk.append(sentence)
            if len(curr_chunk) == chunk_size:
                rows.append(create_chunk())
                curr_chunk = []
        rows.append(create_chunk())  # add the last one
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


def preprocess_labels(y: pd.Series) -> np.ndarray:
    y_codes = pd.Categorical(y).codes
    one_hot = tf.keras.utils.to_categorical(
        y_codes, num_classes=pd.Series(y_codes).unique().size, dtype='float32'
    )
    return np.expand_dims(one_hot, axis=1)


def sentence_level_preprocess(df: pd.DataFrame):
    def helper(X):
        data = num_sentences_based_chucking(X, NUM_OF_SENTENCE_CHUNK)
        res = data["X"].swifter.apply(sentence_level_preprocess_helper)["data"].dropna().reset_index(drop=True)
        return np.vstack(res).reshape((res.size, MAX_LENGTH, EMBEDDING_SIZE)), preprocess_labels(data["y"])

    X_train, X_test, y_train, y_test = train_test_split(df["book_text"], df["author_name"], test_size=TEST_PART)
    return helper(pd.concat([X_train.rename("X"), y_train.rename("y")], axis=1)), \
           helper(pd.concat([X_test.rename("X"), y_test.rename("y")], axis=1))


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

