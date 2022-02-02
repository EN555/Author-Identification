import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from PythonCode.Constants import *
from typing import List
from nltk.corpus import wordnet as wn
from gensim import models
import swifter


def __stem_words(text: str) -> List[str]:
    penn2wn = {'NN': wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
               'VB': wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
               'RB': wn.ADV, 'RBR': wn.ADV, 'RBS': wn.ADV,
               'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ}

    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, penn2wn.get(pos, wn.NOUN)) for word, pos in
            nltk.pos_tag(nltk.word_tokenize(text))]


def bag_of_words(x_train: pd.DataFrame, x_test: pd.DataFrame, **kwargs) -> (pd.DataFrame, pd.DataFrame):
    vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(x_train[TEXT_COLUMN_LABEL])
    x_train = pd.DataFrame(columns=vectorizer.get_feature_names(),
                           data=vectorizer.transform(x_train[TEXT_COLUMN_LABEL]).toarray())
    x_test = pd.DataFrame(columns=vectorizer.get_feature_names(),
                          data=vectorizer.transform(x_test[TEXT_COLUMN_LABEL]).toarray())
    return x_train, x_test


def word_2_vec(x_train: pd.DataFrame, x_test: pd.DataFrame, word2vec_path: str, max_text_len: int, embedding_size: int):
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def get_word2vec(text: str, max_text_len: int, embedding_size: int):
        tmp_encoding = np.append(
            [word2vec.get_vector(word) if word in word_2_vec else [] for word in __stem_words(text)])
        if len(tmp_encoding) > embedding_size * max_text_len:
            encoding = tmp_encoding[:embedding_size * max_text_len]
        else:
            encoding = np.append([tmp_encoding, np.zeros(embedding_size * max_text_len - len(tmp_encoding))])
        return encoding

    return pd.DataFrame(x_train[TEXT_COLUMN_LABEL].swifter.apply(lambda s: get_word2vec(s, max_text_len, embedding_size))), \
           pd.DataFrame(x_test[TEXT_COLUMN_LABEL].swifter.apply(lambda s: get_word2vec(s, max_text_len, embedding_size)))


def aggregative_word2vec(x_train: pd.DataFrame, x_test: pd.DataFrame, word2vec_path: str, embedding_size: int,
                         aggregative_function: callable) -> (pd.DataFrame, pd.DataFrame):

    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def get_aggrigated_word2vec(text: str):
        return aggregative_function(
            [word2vec.get_vector(word) if word in word2vec else np.zeros(embedding_size) for word in __stem_words(text)])

    return pd.DataFrame(x_train[TEXT_COLUMN_LABEL].swifter.apply(get_aggrigated_word2vec)), \
           pd.DataFrame(x_test[TEXT_COLUMN_LABEL].swifter.apply(get_aggrigated_word2vec))
