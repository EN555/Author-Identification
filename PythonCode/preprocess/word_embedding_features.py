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
from sklearn.model_selection import train_test_split

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

DATA_PATH = "../../Data/C50"
OUTPUTS_PATH = "."


def _get_embedding_table(embedding_table):
    if embedding_table is None:
        print("downloading pretrained embedding model.\nthis may take a while...")
        return gensim.downloader.load(GLOVE_MODEL_NAME)
    return embedding_table


def tranform_word(word: str, embedding_table=None) -> Optional[np.ndarray]:
    embedding_table = _get_embedding_table(embedding_table)
    word = re.sub(r'[^a-z]', '', word.lower())
    if word in embedding_table:
        return embedding_table[word]
    return None


def complex_tranform_word(word: str, embedding_table=None) -> Optional[np.ndarray]:
    embedding_table = _get_embedding_table(embedding_table)
    result = tranform_word(word, embedding_table)
    if result is None:
        token_lem = tranform_word(lemmatizer.lemmatize(word), embedding_table)
        token_stem = tranform_word(stemmer.stem(word), embedding_table)
        return token_lem if token_lem is not None else token_stem
    return result


def num_sentences_based_chucking(df: pd.DataFrame, chunk_size: int):
    def create_chunk():
        temp_row = row.copy()
        temp_row[TEXT_COLUMN_NAME] = "".join(curr_chunk)
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


def preprocess_labels(y: pd.Series) -> np.ndarray:
    y_codes = pd.Categorical(y).codes
    one_hot = tf.keras.utils.to_categorical(
        y_codes, num_classes=pd.Series(y_codes).unique().size, dtype='float32'
    )
    return np.expand_dims(one_hot, axis=1)


def sentence_level_preprocess(df: pd.DataFrame, embedding_table):
    def helper(X):
        data = num_sentences_based_chucking(X, NUM_OF_SENTENCE_CHUNK)
        X_pre = np.zeros((data["y"].size, MAX_LENGTH, 300))
        for i, text in enumerate(data["X"]):
            words = nltk.word_tokenize(text)
            for j, word in enumerate(words):
                embedding = tranform_word(word, embedding_table)
                if embedding is not None and j < MAX_LENGTH:
                    X_pre[i, j] = embedding
        return X_pre, preprocess_labels(data["y"])

    X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN_NAME], df[AUTHOR_NAME_COLUMN_NAME],
                                                        test_size=TEST_PART)
    return helper(pd.concat([X_train.rename("X"), y_train.rename("y")], axis=1)), \
           helper(pd.concat([X_test.rename("X"), y_test.rename("y")], axis=1))


def article_level_preprocess(df: pd.DataFrame, embedding_table):
    def helper(X):
        X_pre = np.zeros((X.shape[0], MAX_NUMBER_OF_SENTENCE, EMBEDDING_SIZE), dtype=np.float32)
        for k, text in enumerate(X):
            sentences = nltk.sent_tokenize(text)
            for i, sentence in enumerate(sentences):
                words = nltk.word_tokenize(sentence)
                for j, word in enumerate(words):
                    embedding = tranform_word(word, embedding_table)
                    if embedding is not None and i < MAX_NUMBER_OF_SENTENCE and j < MAX_SENTENCE_LENGTH:
                        X_pre[k, i, j] = embedding
        return X_pre.mean(axis=1, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN_NAME], df[AUTHOR_NAME_COLUMN_NAME],
                                                        test_size=TEST_PART)
    return helper(X_train), helper(X_test), preprocess_labels(y_train), preprocess_labels(y_test)
