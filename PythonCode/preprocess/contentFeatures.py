import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from PythonCode.Constants import *
from typing import List
from nltk.corpus import wordnet as wn
from gensim import models
import swifter


# convert to wordnet system for pos tagging
PENN2WN = {'NN': wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
               'VB': wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
               'RB': wn.ADV, 'RBR': wn.ADV, 'RBS': wn.ADV,
               'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ}


def stem_words(text: str) -> List[str]:
    """
    @return: a list of the words in the given text, lemmitazed
    """
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, PENN2WN.get(pos, wn.NOUN)) for word, pos in
            nltk.pos_tag(nltk.word_tokenize(text))]


def bag_of_words(x_train: pd.DataFrame, x_test: pd.DataFrame, **kwargs) -> (pd.DataFrame, pd.DataFrame):
    """
    @return: bag of words representation for the give data
    """
    vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(x_train[TEXT_COLUMN_LABEL])
    x_train = pd.DataFrame(columns=vectorizer.get_feature_names(),
                           data=vectorizer.transform(x_train[TEXT_COLUMN_LABEL]).toarray())
    x_test = pd.DataFrame(columns=vectorizer.get_feature_names(),
                          data=vectorizer.transform(x_test[TEXT_COLUMN_LABEL]).toarray())
    return x_train, x_test


def word_2_vec(x_train: pd.DataFrame, x_test: pd.DataFrame, word2vec_path: str, max_text_len: int, embedding_size: int):
    """
    @return: a concatenation of the word2vec vectors of each word
    @param word2vec_path: path to the word2vec encodings file
    @param max_text_len: maximum text size in words
    @param embedding_size: size of each word embedding
    """
    # load word2vec
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # get word2vec representation for single text
    def get_word2vec(text: str):
        if len(text) <= 0:
            return np.zeros(embedding_size * max_text_len)
        tmp_encoding = np.concatenate(
            [word2vec.get_vector(word) if word in word2vec else [] for word in stem_words(text)])
        if len(tmp_encoding) > embedding_size * max_text_len:
            encoding = tmp_encoding[:embedding_size * max_text_len]
        else:
            encoding = np.concatenate([tmp_encoding, np.zeros(embedding_size * max_text_len - len(tmp_encoding))])
        return pd.Series(encoding)

    return pd.DataFrame(x_train[TEXT_COLUMN_LABEL].astype(str).apply(get_word2vec)), \
           pd.DataFrame(x_test[TEXT_COLUMN_LABEL].astype(str).apply(get_word2vec))


def aggregative_word2vec(x_train: pd.DataFrame, x_test: pd.DataFrame, word2vec_path: str, embedding_size: int,
                         aggregative_function: callable) -> (pd.DataFrame, pd.DataFrame):
    """
    @return: each text is represented as a single vector of size 'embedding_size',
            attained by applying an aggregative function to all of the word2vec vectors of the words. (i.e mean, sum ...)
    @param word2vec_path: path to the word2vec encodings file
    @param embedding_size: size of each word embedding
    @param aggregative_function: the function to apply (can be np.mean, np,sum etc.)
    """
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # apply to single text
    def get_aggrigated_word2vec(text: str):
        return pd.Series(aggregative_function(
            [word2vec.get_vector(word) if word in word2vec else np.zeros(embedding_size) for word in stem_words(text)], axis=0))

    return pd.DataFrame(x_train[TEXT_COLUMN_LABEL].swifter.apply(get_aggrigated_word2vec)), \
           pd.DataFrame(x_test[TEXT_COLUMN_LABEL].swifter.apply(get_aggrigated_word2vec))
