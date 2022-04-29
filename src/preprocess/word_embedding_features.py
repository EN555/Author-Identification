import tensorflow as tf
import numpy as np
import nltk
import pandas as pd
import gensim.downloader
import gensim
import re
from typing import Optional
from src.config.Constants import *
from sklearn.model_selection import train_test_split

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

DATA_PATH = "../../data/C50"
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


def preprocess_labels(y: pd.Series) -> np.ndarray:
    y_codes = pd.Categorical(y).codes
    one_hot = tf.keras.utils.to_categorical(
        y_codes, num_classes=pd.Series(y_codes).unique().size, dtype='float32'
    )
    return np.expand_dims(one_hot, axis=1)


def sentence_level_preprocess(X, embedding_table=None):
    X_pre = np.zeros((MAX_LENGTH, EMBEDDING_SIZE))
    words = nltk.word_tokenize(X)
    for j, word in enumerate(words):
        embedding = tranform_word(word, embedding_table)
        if embedding is not None and j < MAX_LENGTH:
            X_pre[j] = embedding
    return X_pre


def article_level_preprocess(df: pd.DataFrame, embedding_table=None):
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

