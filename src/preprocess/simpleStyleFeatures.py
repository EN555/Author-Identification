import string
from nltk.corpus import stopwords
import pandas as pd
from src.config.Constants import *
import nltk


def simple_style_features_extraction(x_train: pd.DataFrame, x_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    @return: a representation of the data using simple stylistic features defined in this file
    """

    def simple_style_features_extraction_helper(x: pd.DataFrame) -> pd.DataFrame:
        x = pd.concat(
            [pos_count(x),
             stop_words_count(x),
             avg_word_len(x),
             avg_sentence_len(x),
             punctuation_marks_count(x)], axis=1)
        return x

    return simple_style_features_extraction_helper(x_train), simple_style_features_extraction_helper(x_test)


def avg_word_len(df: pd.DataFrame) -> pd.DataFrame:
    avg_word_len = df[TEXT_COLUMN_NAME].astype(str).swifter.apply(
        lambda s: pd.Series(nltk.word_tokenize(s)).map(len).mean()).rename("avg_word_len")
    return pd.DataFrame(avg_word_len)


def avg_sentence_len(df: pd.DataFrame) -> pd.DataFrame:
    sentence_count = df[TEXT_COLUMN_NAME].astype(str).swifter.apply(
        lambda text: pd.Series(nltk.sent_tokenize(text)).map(
            lambda sent: len(nltk.word_tokenize(sent))).mean()).rename("avg_sentence_len")

    return pd.DataFrame(sentence_count)


def punctuation_marks_count(df: pd.DataFrame) -> pd.DataFrame:
    to_return = pd.DataFrame()
    for mark in list(string.punctuation):
        to_return[mark] = df[TEXT_COLUMN_NAME].astype(str).swifter.apply(lambda s: s.count(mark) / len(s))
    return to_return


def stop_words_count(df: pd.DataFrame) -> pd.DataFrame:
    def helper(text: str, curr_word: str):
        words = nltk.word_tokenize(text)
        return words.count(curr_word) / len(words)

    to_return = pd.DataFrame()
    for word in list(stopwords.words('english')):
        to_return[word] = df[TEXT_COLUMN_NAME].astype(str).swifter.apply(lambda s: helper(s,word))
    return to_return


def pos_count(df: pd.DataFrame) -> pd.DataFrame:
    def group_pos(tag):
        groups = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "verb": ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                  "adverb": ['RB', 'RBR', 'RBS'], "adjective": ['JJ', 'JJR', 'JJS']}
        for key, group in groups.items():
            if tag in group:
                return key
        return None

    features = df[TEXT_COLUMN_NAME].astype(str).swifter.apply(
        lambda s: pd.Series([x[1] for x in nltk.pos_tag(nltk.word_tokenize(s))], dtype=str).
            apply(group_pos).value_counts(normalize=True).copy())
    features = features.fillna(0)
    return features


# number of time uses comma
def average_comma_uses(data) -> pd.DataFrame:
    data["sentence_len"] = data["Text"].swifter \
        .apply(lambda text: pd.Series(nltk.sent_tokenize(text)).map(
        lambda sent: pd.Series(sent).value_counts()[","] / len(nltk.word_tokenize(sent))).mean())
    return data


# average of sentence
def average_sentence_size(data) -> pd.DataFrame:
    data["sentence_len"] = data["Text"].swifter \
        .apply(lambda text: pd.Series(nltk.sent_tokenize(text)).map(lambda sent: len(nltk.word_tokenize(sent))).mean())
    return data
